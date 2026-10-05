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
from business_entity_resolution.blocking.candidate_generator import union_and_rank
from business_entity_resolution.features.build_feature_matrix import build_feature_matrix
from business_entity_resolution.pipeline.common import load_normalize_block
from business_entity_resolution.models.threshold import predict_matches
from business_entity_resolution.evaluation.report import full_validation_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    t_start = time.time()
    
    logger.info("[Step 1/6] Loading S1 and GT (Expected ~15s)...")
    s1 = loader.load_source(config.TRAIN_FILES["S1"])
    gt = loader.load_ground_truth(config.TRAIN_GROUND_TRUTH)
    gt_long = loader.explode_ground_truth(gt)

    sampled_ids = stratified_sample_s1(s1, gt, 50000, seed=42)
    s1 = s1[s1["entity_id"].isin(sampled_ids)].reset_index(drop=True)
    del gt; gc.collect()
    logger.info(f" -> S1 sampling complete (Took {time.time()-t_start:.1f}s)")

    logger.info("[Step 2/6] Normalizing S1 (Expected ~15s)...")
    t_norm = time.time()
    s1n = normalize_source(s1)
    s1_ids = s1n["entity_id"].tolist()
    del s1; gc.collect()
    logger.info(f" -> S1 normalization complete (Took {time.time()-t_norm:.1f}s)")

    logger.info("[Step 3/6] Blocking candidates across S2 and S3 (Expected ~8-10m)...")
    t_block = time.time()
    logger.info("   -> Blocking S2...")
    cand2, s2_subset = load_normalize_block(s1n, config.TRAIN_FILES["S2"], "S2", config.DEFAULT_CONFIG.blocking)
    logger.info(f"   -> S2 Blocking complete (Took {time.time()-t_block:.1f}s). Blocking S3...")
    
    t_block3 = time.time()
    cand3, s3_subset = load_normalize_block(s1n, config.TRAIN_FILES["S3"], "S3", config.DEFAULT_CONFIG.blocking)
    logger.info(f"   -> S3 Blocking complete (Took {time.time()-t_block3:.1f}s). Unioning...")
    
    candidates = union_and_rank(pd.concat([cand2, cand3], ignore_index=True), config.DEFAULT_CONFIG.blocking.max_candidates_per_s1)
    del cand2, cand3; gc.collect()
    logger.info(f" -> Total blocking complete. {len(candidates)} candidates generated (Took {time.time()-t_block:.1f}s total)")

    logger.info("[Step 4/6] Feature engineering (Expected ~4-5m)...")
    t_feat = time.time()
    others_n = pd.concat([s2_subset, s3_subset], ignore_index=True)
    del s2_subset, s3_subset; gc.collect()

    X, meta = build_feature_matrix(candidates[["source1_entity_id", "candidate_entity_id", "score"]], s1n, others_n, config.DEFAULT_CONFIG.features)
    del others_n, candidates, s1n; gc.collect()
    logger.info(f" -> Feature engineering complete (Took {time.time()-t_feat:.1f}s)")

    logger.info("[Step 5/6] Loading artifacts and Predicting (Expected ~30s)...")
    t_pred = time.time()
    model = joblib.load(config.ARTIFACTS_DIR / "lgbm_model.joblib")
    features = json.loads((config.ARTIFACTS_DIR / "feature_columns.json").read_text())
    thresh_info = json.loads((config.ARTIFACTS_DIR / "threshold.json").read_text())
    tau = thresh_info["tau"]

    X = X[features]
    proba = model.predict_proba(X)[:, 1]
    meta["proba"] = proba
    logger.info(f" -> Predictions generated (Took {time.time()-t_pred:.1f}s)")

    logger.info("[Step 6/6] Evaluating metrics...")
    t_eval = time.time()
    val_predictions = predict_matches(meta, tau, apply_reverse_constraint=True)

    gt_subset = gt_long[gt_long["source1_entity_id"].isin(sampled_ids)]
    gt_dict = gt_subset.groupby("source1_entity_id")["matched_entity_id"].apply(set).to_dict()
    gt_dict = {sid: gt_dict.get(sid, set()) for sid in s1_ids}

    report = full_validation_report(val_predictions, gt_dict, beta=config.DEFAULT_CONFIG.threshold.beta)
    logger.info(f" -> Evaluation complete (Took {time.time()-t_eval:.1f}s)")

    print("\n" + "="*50)
    print("--- FULL 500,001 TRAIN-SET EVALUATION (Extrapolated from 50k representative sample) ---")
    print(f"Threshold used: {tau:.4f}")
    print(f"Macro F0.5: {report['macro_f_beta']:.4f}")
    print(f"Macro Precision: {report['macro_precision']:.4f}")
    print(f"Macro Recall: {report['macro_recall']:.4f}")
    print(f"Singleton Accuracy: {report['singleton_accuracy']:.2%}")
    print(f"Total True Pairs (Extrapolated): {report.get('total_true_pairs', 0) * 10}")
    print(f"Total Predicted (Extrapolated): {report.get('total_predicted_pairs', 0) * 10}")
    print("="*50 + "\n")

    with open("scratch/eval_train_results.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
