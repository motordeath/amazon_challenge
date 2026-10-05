import gc
import json
import logging
import os
import time
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Enforce strict CPU thread safety to prevent deadlocks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
torch.set_num_threads(1)

import sys
sys.path.append(str(Path("code/business_entity_resolution/src").resolve()))

from business_entity_resolution import config
from business_entity_resolution.data import loader
from business_entity_resolution.pipeline.sampling import stratified_sample_s1
from business_entity_resolution.normalization.run_normalize import normalize_source
from business_entity_resolution.blocking.candidate_generator import generate_candidates_for_pair, union_and_rank
from business_entity_resolution.blocking.recall_eval import blocking_recall
from business_entity_resolution.features.build_feature_matrix import build_feature_matrix
from business_entity_resolution.models.threshold import predict_matches
from business_entity_resolution.evaluation.report import full_validation_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = config.ARTIFACTS_DIR / "embedding_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def concat_text(row):
    name = str(row.get("name_normalized", "")) if pd.notna(row.get("name_normalized")) else ""
    addr = str(row.get("address_normalized", "")) if pd.notna(row.get("address_normalized")) else ""
    return (name + " " + addr).strip()

def run_experiment():
    logger.info("Loading S1 and GT...")
    s1_raw = loader.load_source(config.TRAIN_FILES["S1"])
    gt = loader.load_ground_truth(config.TRAIN_GROUND_TRUTH)
    gt_long = loader.explode_ground_truth(gt)

    sampled_ids = stratified_sample_s1(s1_raw, gt, 500001, seed=42)
    s1 = s1_raw[s1_raw["entity_id"].isin(sampled_ids)].reset_index(drop=True)
    gt_subset = gt_long[gt_long["source1_entity_id"].isin(sampled_ids)]
    
    logger.info("Normalizing S1...")
    s1n = normalize_source(s1)
    del s1_raw, s1; gc.collect()
    s1n["text"] = s1n.apply(concat_text, axis=1)

    logger.info("Loading S2 and S3...")
    s2n = normalize_source(loader.load_source(config.TRAIN_FILES["S2"]))
    s2n["text"] = s2n.apply(concat_text, axis=1)
    
    s3n = normalize_source(loader.load_source(config.TRAIN_FILES["S3"]))
    s3n["text"] = s3n.apply(concat_text, axis=1)

    logger.info("Generating standard baseline candidates...")
    base_cand2 = generate_candidates_for_pair(s1n, s2n, "S2", config.DEFAULT_CONFIG.blocking)
    base_cand3 = generate_candidates_for_pair(s1n, s3n, "S3", config.DEFAULT_CONFIG.blocking)
    baseline_uncapped = pd.concat([base_cand2, base_cand3], ignore_index=True)
    baseline_capped = union_and_rank(baseline_uncapped, config.DEFAULT_CONFIG.blocking.max_candidates_per_s1)
    del base_cand2, base_cand3; gc.collect()

    # Determine triggers
    cand_counts = baseline_capped.groupby("source1_entity_id").size().to_dict()
    
    logger.info("Initializing MiniLM embedding model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    target_pool = pd.concat([
        s2n[["entity_id", "country", "text"]], 
        s3n[["entity_id", "country", "text"]]
    ], ignore_index=True)
    
    emb_path = CACHE_DIR / "target_embs.npy"
    if emb_path.exists():
        logger.info("Loading cached target embeddings...")
        target_embs = np.load(emb_path)
    else:
        logger.info("Encoding 2.8M reference records (This takes ~4h on CPU without GPU)...")
        target_embs = model.encode(target_pool["text"].tolist(), batch_size=256, show_progress_bar=True, normalize_embeddings=True)
        np.save(emb_path, target_embs)
    
    target_embs = target_embs.astype(np.float32)
    target_pool["emb_idx"] = np.arange(len(target_pool))

    logger.info("Building Country-Partitioned FAISS Indexes...")
    indices = {}
    target_entity_maps = {}
    for country in s1n["country"].dropna().unique():
        c_targets = target_pool[target_pool["country"] == country]
        if c_targets.empty: continue
        idx = faiss.IndexFlatIP(384)
        idx.add(target_embs[c_targets["emb_idx"].values])
        indices[country] = idx
        target_entity_maps[country] = c_targets["entity_id"].values

    logger.info("Encoding 500,001 S1 queries...")
    s1_emb_path = CACHE_DIR / "s1_embs.npy"
    if s1_emb_path.exists():
        s1_embs = np.load(s1_emb_path)
    else:
        s1_embs = model.encode(s1n["text"].tolist(), batch_size=256, show_progress_bar=True, normalize_embeddings=True)
        np.save(s1_emb_path, s1_embs)
    
    s1_embs = s1_embs.astype(np.float32)
    s1n["emb_idx"] = np.arange(len(s1n))

    lgbm_model = joblib.load(config.ARTIFACTS_DIR / "lgbm_model.joblib")
    features = json.loads((config.ARTIFACTS_DIR / "feature_columns.json").read_text())
    
    # We will compute feature matrix for reference others
    others_n = pd.concat([s2n, s3n], ignore_index=True)

    triggers = [1, 3, 5, 10]
    results = []

    for t in triggers:
        logger.info(f"\n{'='*40}\nRunning Trigger < {t}\n{'='*40}")
        t0 = time.time()
        
        # 1. Identify S1s triggering the fallback
        s1n["cand_count"] = s1n["entity_id"].map(lambda x: cand_counts.get(x, 0))
        triggered_s1 = s1n[s1n["cand_count"] < t].copy()
        n_triggered = len(triggered_s1)
        logger.info(f"Triggered for {n_triggered} S1 entities")

        emb_cands = []
        if n_triggered > 0:
            for country, idx in indices.items():
                c_s1 = triggered_s1[triggered_s1["country"] == country]
                if c_s1.empty: continue
                
                D, I = idx.search(s1_embs[c_s1["emb_idx"].values], k=10)
                q_maps = c_s1["entity_id"].values
                t_maps = target_entity_maps[country]
                
                for i, q_id in enumerate(q_maps):
                    for j, match_idx in enumerate(I[i]):
                        if match_idx >= 0:
                            emb_cands.append({
                                "source1_entity_id": q_id,
                                "candidate_entity_id": t_maps[match_idx],
                                "score": float(D[i][j]),
                                "channel": "embedding"
                            })
        
        emb_df = pd.DataFrame(emb_cands) if emb_cands else pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score", "channel"])
        n_emb_cands = len(emb_df)
        
        union_uncapped = pd.concat([baseline_uncapped, emb_df], ignore_index=True)
        # Drop duplicates prioritizing existing scores (or custom resolution)
        union_uncapped = union_uncapped.drop_duplicates(["source1_entity_id", "candidate_entity_id"], keep="first")
        
        final_capped = union_and_rank(union_uncapped, 25)
        
        n_unique_total = len(final_capped)
        overlap = len(emb_df) - (len(union_uncapped) - len(baseline_uncapped))
        
        rec_before = blocking_recall(union_uncapped, gt_subset)
        rec_after = blocking_recall(final_capped, gt_subset)
        
        add_true_pairs = rec_after['retrieved_true_pairs'] - 832216  # baseline retrieved
        
        logger.info("Feature engineering and scoring for downstream model...")
        X, meta = build_feature_matrix(final_capped[["source1_entity_id", "candidate_entity_id", "score"]], s1n, others_n, config.DEFAULT_CONFIG.features)
        
        X = X[features]
        meta["proba"] = lgbm_model.predict_proba(X)[:, 1]
        
        val_predictions = predict_matches(meta, 0.900, apply_reverse_constraint=True)
        gt_dict = {sid: set(grp["matched_entity_id"]) for sid, grp in gt_subset.groupby("source1_entity_id")}
        
        report = full_validation_report(val_predictions, gt_dict, beta=0.5)
        
        rt = time.time() - t0
        
        results.append({
            "trigger": f"<{t}",
            "s1_triggered": n_triggered,
            "emb_cands": n_emb_cands,
            "unique_total": n_unique_total,
            "overlap": overlap,
            "recall_before": rec_before['blocking_recall'],
            "recall_after": rec_after['blocking_recall'],
            "add_true_pairs": add_true_pairs,
            "macro_f05": report['macro_f_beta'],
            "precision": report['macro_precision'],
            "recall": report['macro_recall'],
            "singleton_acc": report['singleton_accuracy'],
            "runtime": rt
        })
        
        logger.info(f"Trigger <{t} completed. F0.5: {report['macro_f_beta']:.4f}")

    res_df = pd.DataFrame(results)
    print("\n\n=== EMBEDDING RESCUE CONFIGURATION RESULTS ===")
    print(res_df.to_string(index=False))
    res_df.to_csv(CACHE_DIR / "embedding_rescue_results.csv", index=False)

if __name__ == "__main__":
    run_experiment()
