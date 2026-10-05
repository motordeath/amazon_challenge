import gc
import json
import logging
import sys
import time
import psutil
import os
from pathlib import Path
import pandas as pd
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import faiss

sys.path.append(str(Path("code/business_entity_resolution/src").resolve()))
from business_entity_resolution import config
from business_entity_resolution.data import loader
from business_entity_resolution.pipeline.sampling import stratified_sample_s1
from business_entity_resolution.normalization.run_normalize import normalize_source
from business_entity_resolution.blocking.candidate_generator import generate_candidates_for_pair, union_and_rank
from business_entity_resolution.blocking.recall_eval import blocking_recall

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def get_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def concat_text(row):
    name = str(row.get("name_normalized", "")) if pd.notna(row.get("name_normalized")) else ""
    addr = str(row.get("address_normalized", "")) if pd.notna(row.get("address_normalized")) else ""
    return (name + " " + addr).strip()

def run_ablation():
    t_start = time.time()
    
    logger.info("Loading GT and generating 500,001 S1 sample...")
    s1 = loader.load_source(config.TRAIN_FILES["S1"])
    gt = loader.load_ground_truth(config.TRAIN_GROUND_TRUTH)
    gt_long = loader.explode_ground_truth(gt)
    
    sampled_ids = stratified_sample_s1(s1, gt, 500001, seed=42)
    s1 = s1[s1["entity_id"].isin(sampled_ids)].reset_index(drop=True)
    gt_subset = gt_long[gt_long["source1_entity_id"].isin(sampled_ids)]
    
    logger.info("Normalizing S1...")
    s1n = normalize_source(s1)
    del s1; gc.collect()
    
    s1n["text"] = s1n.apply(concat_text, axis=1)
    
    logger.info("Loading and normalizing S2/S3...")
    s2_raw = loader.load_source(config.TRAIN_FILES["S2"])
    s2n = normalize_source(s2_raw)
    del s2_raw; gc.collect()
    s2n["text"] = s2n.apply(concat_text, axis=1)
    
    s3_raw = loader.load_source(config.TRAIN_FILES["S3"])
    s3n = normalize_source(s3_raw)
    del s3_raw; gc.collect()
    s3n["text"] = s3n.apply(concat_text, axis=1)
    
    logger.info("Generating baseline candidates (UNCAPPED)...")
    base_cand2 = generate_candidates_for_pair(s1n, s2n, "S2", config.DEFAULT_CONFIG.blocking)
    base_cand3 = generate_candidates_for_pair(s1n, s3n, "S3", config.DEFAULT_CONFIG.blocking)
    
    baseline_uncapped = pd.concat([base_cand2, base_cand3], ignore_index=True)
    del base_cand2, base_cand3; gc.collect()
    
    # Pre-calculate target texts for LSA
    target_pool = pd.concat([s2n[["entity_id", "country", "text"]], s3n[["entity_id", "country", "text"]]], ignore_index=True)
    target_pool["emb_idx"] = np.arange(len(target_pool))
    
    # Clean up massive text frames, we only need 'text' for LSA
    s1_texts = s1n["text"].tolist()
    target_texts = target_pool["text"].tolist()
    
    results = []

    def evaluate_lsa(ngram_range, n_components):
        logger.info(f"\n--- Running Ablation: N-Gram {ngram_range}, Dim {n_components} ---")
        t_conf = time.time()
        
        logger.info("TF-IDF vectorization...")
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=ngram_range, min_df=2, sublinear_tf=True)
        # Fit on targets
        T_targets = vectorizer.fit_transform(target_texts)
        T_queries = vectorizer.transform(s1_texts)
        
        logger.info("LSA (TruncatedSVD)...")
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        target_embs = svd.fit_transform(T_targets).astype(np.float32)
        query_embs = svd.transform(T_queries).astype(np.float32)
        
        # L2 norm for cosine similarity
        faiss.normalize_L2(target_embs)
        faiss.normalize_L2(query_embs)
        
        logger.info("FAISS Indexing and Querying (Top-20 per country)...")
        lsa_cands = []
        for country in s1n["country"].dropna().unique():
            c_targets = target_pool[target_pool["country"] == country]
            if c_targets.empty: continue
            
            c_s1 = s1n[s1n["country"] == country]
            if c_s1.empty: continue
            
            idx = faiss.IndexFlatIP(n_components)
            c_target_embs = target_embs[c_targets["emb_idx"].values]
            idx.add(c_target_embs)
            
            c_q_embs = query_embs[c_s1.index.values]
            D, I = idx.search(c_q_embs, k=20)
            
            t_maps = c_targets["entity_id"].values
            q_maps = c_s1["entity_id"].values
            
            for i, q_id in enumerate(q_maps):
                for j, match_idx in enumerate(I[i]):
                    if match_idx >= 0 and D[i][j] > 0.3:  # minimal similarity cutoff
                        lsa_cands.append({
                            "source1_entity_id": q_id,
                            "candidate_entity_id": t_maps[match_idx],
                            "score": float(D[i][j]),
                            "channel": "lsa"
                        })
        
        lsa_df = pd.DataFrame(lsa_cands)
        if lsa_df.empty:
            lsa_df = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score", "channel"])
            
        # 1. LSA Channel Recall (Isolation)
        lsa_recall = blocking_recall(lsa_df, gt_subset)
        
        # 2. Union UNCAPPED
        union_uncapped = pd.concat([baseline_uncapped, lsa_df], ignore_index=True)
        # Drop strict duplicates (same pair, different channel/score). Keep highest score.
        union_uncapped = union_uncapped.sort_values("score", ascending=False).drop_duplicates(["source1_entity_id", "candidate_entity_id"], keep="first")
        
        union_recall = blocking_recall(union_uncapped, gt_subset)
        cand_before = len(union_uncapped)
        
        # 3. Apply 25 cap
        final_capped = union_and_rank(union_uncapped, 25)
        final_recall = blocking_recall(final_capped, gt_subset)
        cand_after = len(final_capped)
        
        rt = time.time() - t_conf
        peak_ram = get_rss_mb()
        
        logger.info(f"Done. Final Recall: {final_recall['blocking_recall']:.4f}")
        
        results.append({
            "ngram": str(ngram_range),
            "dim": n_components,
            "lsa_channel_recall": lsa_recall['blocking_recall'],
            "union_uncapped_recall": union_recall['blocking_recall'],
            "final_capped_recall": final_recall['blocking_recall'],
            "cand_before": cand_before,
            "cand_after": cand_after,
            "runtime_s": rt,
            "peak_ram_mb": peak_ram
        })
        
        # Free memory aggressively
        del vectorizer, T_targets, T_queries, svd, target_embs, query_embs, lsa_df, union_uncapped, final_capped, lsa_cands
        gc.collect()

    # --- EXPERIMENT A ---
    ngrams_to_test = [(2,4), (3,5), (3,6), (2,5), (2,6)]
    for ng in ngrams_to_test:
        evaluate_lsa(ng, 64)
        
    # --- EXPERIMENT B ---
    # Find best ngram from Exp A by final_capped_recall
    best_exp_a = max(results, key=lambda x: x["final_capped_recall"])
    best_ng = eval(best_exp_a["ngram"])
    
    dims_to_test = [32, 128, 256] # 64 already tested
    for d in dims_to_test:
        evaluate_lsa(best_ng, d)
        
    res_df = pd.DataFrame(results)
    print("\n\n=== ABLATION RESULTS ===")
    print(res_df.to_string(index=False))
    res_df.to_csv("scratch/lsa_ablation_results.csv", index=False)

if __name__ == "__main__":
    run_ablation()
