import sys
import gc
import json
import logging
import time
from pathlib import Path
import pandas as pd
import joblib

sys.path.append(str(Path("code/business_entity_resolution/src").resolve()))
from business_entity_resolution import config
from business_entity_resolution.data import loader
from business_entity_resolution.pipeline.sampling import stratified_sample_s1
from business_entity_resolution.normalization.run_normalize import normalize_source
from business_entity_resolution.blocking.candidate_generator import generate_candidates_for_pair, union_and_rank
from business_entity_resolution.features.build_feature_matrix import build_feature_matrix
from business_entity_resolution.models.threshold import predict_matches
from business_entity_resolution.evaluation.report import full_validation_report
from business_entity_resolution.models.train_lgbm import split_all_s1_ids
from business_entity_resolution.pipeline.common import load_normalize_block

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

def run_ablation():
    logger.info("Loading S1 and GT...")
    s1 = loader.load_source(config.TRAIN_FILES["S1"])
    gt = loader.load_ground_truth(config.TRAIN_GROUND_TRUTH)
    gt_long = loader.explode_ground_truth(gt)

    sampled_ids = stratified_sample_s1(s1, gt, 500001, seed=42)
    s1 = s1[s1["entity_id"].isin(sampled_ids)].reset_index(drop=True)

    logger.info("Normalizing S1...")
    s1n = normalize_source(s1)
    del s1; gc.collect()
    
    # We only need the validation split for this analysis
    cfg = config.DEFAULT_CONFIG
    _, val_ids = split_all_s1_ids(s1n["entity_id"], n_splits=5, fold=0, seed=cfg.random_seed)
    val_ids_set = set(val_ids)
    s1n_val = s1n[s1n["entity_id"].isin(val_ids_set)].reset_index(drop=True)
    
    gt_subset = gt_long[gt_long["source1_entity_id"].isin(val_ids_set)]
    gt_dict = {sid: set(grp["matched_entity_id"]) for sid, grp in gt_subset.groupby("source1_entity_id")}
    gt_dict = {sid: gt_dict.get(sid, set()) for sid in val_ids}

    logger.info(f"Blocking S2/S3 for {len(val_ids)} validation queries...")
    cand2, s2_subset = load_normalize_block(s1n_val, config.TRAIN_FILES["S2"], "S2", cfg.blocking)
    cand3, s3_subset = load_normalize_block(s1n_val, config.TRAIN_FILES["S3"], "S3", cfg.blocking)
    candidates = union_and_rank(pd.concat([cand2, cand3], ignore_index=True), cfg.blocking.max_candidates_per_s1)
    del cand2, cand3; gc.collect()

    logger.info("Feature engineering...")
    others_n = pd.concat([s2_subset, s3_subset], ignore_index=True)
    del s2_subset, s3_subset; gc.collect()

    X, meta = build_feature_matrix(candidates[["source1_entity_id", "candidate_entity_id", "score"]], s1n_val, others_n, cfg.features)
    del others_n, candidates, s1n_val; gc.collect()

    logger.info("Loading model...")
    model = joblib.load(config.ARTIFACTS_DIR / "lgbm_model.joblib")
    features = json.loads((config.ARTIFACTS_DIR / "feature_columns.json").read_text())
    
    X = X[features]
    proba = model.predict_proba(X)[:, 1]
    meta["proba"] = proba

    thresholds = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
    results = []

    logger.info("Running ablation...")
    for tau in thresholds:
        val_predictions = predict_matches(meta, tau, apply_reverse_constraint=True)
        report = full_validation_report(val_predictions, gt_dict, beta=cfg.threshold.beta)
        
        # Calculate predicted match count explicitly
        predicted_count = sum(len(matches) for matches in val_predictions.values())
        
        results.append({
            "threshold": tau,
            "macro_f05": report['macro_f_beta'],
            "precision": report['macro_precision'],
            "recall": report['macro_recall'],
            "singleton_acc": report['singleton_accuracy'],
            "predicted_count": predicted_count
        })

    res_df = pd.DataFrame(results)
    print("\n\n=== THRESHOLD ABLATION RESULTS ===")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_ablation()
