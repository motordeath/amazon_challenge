"""End-to-end training pipeline: load -> [sample] -> normalize -> block -> feature -> train -> threshold.

Saves artifacts (model, feature list, threshold, blocking config) under
``src/business_entity_resolution/artifacts/`` so ``inference_pipeline`` can
reproduce inference without retraining.

``s1_sample_size`` is a *training/validation-mode-only* convenience: it
stratified-subsamples the S1 query side (never S2/S3) for faster iteration
on memory-constrained hardware. ``inference_pipeline`` never subsamples —
every test S1 entity is always scored.
"""
from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil

from .. import config
from ..blocking.candidate_generator import union_and_rank
from ..blocking.recall_eval import blocking_recall
from ..data import loader
from ..evaluation.report import full_validation_report
from ..features.build_feature_matrix import build_feature_matrix
from ..models.threshold import predict_matches, tune_global_threshold
from ..models.train_lgbm import feature_importance, split_all_s1_ids, train_lgbm
from ..normalization.run_normalize import normalize_source
from .common import NORMALIZED_KEEP_COLS, load_normalize_block
from .labeling import label_candidates
from .sampling import sample_report, stratified_sample_s1

logger = logging.getLogger(__name__)


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def run_train_pipeline(
    cfg: config.PipelineConfig = config.DEFAULT_CONFIG,
    nrows: int | None = None,
    s1_sample_size: int | None = None,
    s1_sample_seed: int = 42,
) -> dict:
    t_start = time.time()
    report: dict = {}

    logger.info("loading S1 + ground truth (nrows=%s)", nrows)
    s1 = loader.load_source(config.TRAIN_FILES["S1"], nrows=nrows)
    gt = loader.load_ground_truth(config.TRAIN_GROUND_TRUTH, nrows=nrows)

    if s1_sample_size is not None:
        sampled_ids = stratified_sample_s1(s1, gt, s1_sample_size, seed=s1_sample_seed)
        report["sample_distribution"] = sample_report(s1, gt, sampled_ids)
        s1 = s1[s1["entity_id"].isin(sampled_ids)].reset_index(drop=True)
        gt = gt[gt["source1_entity_id"].isin(sampled_ids)].reset_index(drop=True)
    report["n_s1_sampled"] = len(s1)
    logger.info("training on %d S1 entities%s", len(s1), f" (stratified sample, seed={s1_sample_seed})" if s1_sample_size else " (full)")

    gt_long = loader.explode_ground_truth(gt)
    del gt
    s1n = normalize_source(s1)
    del s1
    s1n = s1n[NORMALIZED_KEEP_COLS].copy()  # drop raw business_name/business_address — see common.py
    gc.collect()

    logger.info("blocking (S2 and S3 processed sequentially to bound peak memory)")
    cand2, s2_subset = load_normalize_block(s1n, config.TRAIN_FILES["S2"], "S2", cfg.blocking, nrows=nrows)
    cand3, s3_subset = load_normalize_block(s1n, config.TRAIN_FILES["S3"], "S3", cfg.blocking, nrows=nrows)
    candidates = union_and_rank(pd.concat([cand2, cand3], ignore_index=True), cfg.blocking.max_candidates_per_s1)
    del cand2, cand3
    gc.collect()
    report["n_candidate_pairs"] = len(candidates)

    recall_report = blocking_recall(candidates, gt_long)
    report["blocking_recall"] = recall_report
    logger.info("blocking recall: %.4f (%d/%d true pairs) | avg candidates/s1=%.1f",
                recall_report["blocking_recall"], recall_report["retrieved_true_pairs"],
                recall_report["total_true_pairs"], recall_report["avg_candidates_per_s1"])

    logger.info("labeling")
    labeled = label_candidates(candidates, gt_long)
    del candidates  # superseded by `labeled`; holding both doubles a multi-million-row frame for no reason
    n_pos = int((labeled["label"] == 1).sum())
    n_neg = int((labeled["label"] == 0).sum())
    report["n_positive_pairs"] = n_pos
    report["n_negative_pairs"] = n_neg
    logger.info("labeled pairs: positive=%d negative=%d (ratio 1:%.1f)", n_pos, n_neg, n_neg / max(n_pos, 1))
    y = labeled["label"].to_numpy()
    small_labeled = labeled[["source1_entity_id", "candidate_entity_id", "score"]]
    del labeled  # y and small_labeled are all that's needed from here on
    gc.collect()

    logger.info("feature engineering")
    others_n = pd.concat([s2_subset, s3_subset], ignore_index=True)
    del s2_subset, s3_subset
    gc.collect()
    rss_before = _rss_mb()
    X, meta = build_feature_matrix(small_labeled, s1n, others_n, cfg.features)
    del others_n, small_labeled  # X/meta are the only outputs still needed; the multi-million-row
    gc.collect()                 # inputs that produced them must not outlive the call that consumed them
    rss_after = _rss_mb()
    report["feature_engineering_rss_mb"] = {"before": round(rss_before, 1), "after": round(rss_after, 1), "delta": round(rss_after - rss_before, 1)}
    logger.info("feature engineering RSS: before=%.0fMB after=%.0fMB delta=%.0fMB", rss_before, rss_after, rss_after - rss_before)

    groups = meta["source1_entity_id"]

    logger.info("train/val split (fold split over the FULL S1 id space, not just ids with candidates)")
    train_ids, val_ids = split_all_s1_ids(s1n["entity_id"], n_splits=5, fold=0, seed=cfg.random_seed)
    train_mask = groups.isin(train_ids).to_numpy()
    val_mask = groups.isin(val_ids).to_numpy()
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    meta_val = meta[val_mask].reset_index(drop=True)

    logger.info("training LightGBM on %d rows (%d positive)", len(X_train), int(y_train.sum()))
    t_train = time.time()
    model = train_lgbm(X_train, y_train, X_val, y_val, cfg.model)
    report["training_time_sec"] = round(time.time() - t_train, 1)
    report["feature_importance"] = feature_importance(model, list(X.columns)).to_dict("records")
    logger.info("LightGBM training took %.1fs", report["training_time_sec"])

    val_s1_ids = sorted(val_ids)
    gt_val = gt_long[gt_long["source1_entity_id"].isin(val_ids)]
    gt_val_dict = gt_val.groupby("source1_entity_id")["matched_entity_id"].apply(set).to_dict()
    gt_val_dict = {sid: gt_val_dict.get(sid, set()) for sid in val_s1_ids}

    proba_val = model.predict_proba(X_val)[:, 1]
    scored_val = meta_val.copy()
    scored_val["proba"] = proba_val

    logger.info("tuning decision threshold")
    thresh_report = tune_global_threshold(scored_val, gt_val_dict, val_s1_ids, cfg.threshold)
    report["threshold"] = thresh_report["best"]
    logger.info("best threshold=%.3f macro_F0.5=%.4f", thresh_report["best"]["tau"], thresh_report["best"]["macro_f_beta"])

    val_predictions = predict_matches(scored_val, thresh_report["best"]["tau"], apply_reverse_constraint=True)
    full_report = full_validation_report(val_predictions, gt_val_dict, beta=cfg.threshold.beta)
    report["validation"] = full_report
    logger.info(
        "VALIDATION: macro_F0.5=%.4f macro_P=%.4f macro_R=%.4f | singletons n=%d acc=%.4f | multi-match n=%d",
        full_report["macro_f_beta"], full_report["macro_precision"], full_report["macro_recall"],
        full_report["singleton_count"], full_report["singleton_accuracy"], full_report["multi_match_count"],
    )

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.ARTIFACTS_DIR / "lgbm_model.joblib")
    (config.ARTIFACTS_DIR / "feature_columns.json").write_text(json.dumps(list(X.columns)))
    (config.ARTIFACTS_DIR / "threshold.json").write_text(json.dumps(thresh_report["best"]))
    (config.ARTIFACTS_DIR / "train_report.json").write_text(json.dumps(report, default=str, indent=2))

    report["runtime_sec"] = time.time() - t_start
    logger.info("train pipeline done in %.1fs", report["runtime_sec"])
    return report
