"""End-to-end inference pipeline: load test data -> block -> feature -> score -> decide -> write outputs.

``candidate_pairs.tsv`` is written from the exact same candidate set the
model scores (post blocking-union, pre-threshold) — the task spec requires
this to be the *last* filtering stage before the model, not an earlier pass.
``matching_results.tsv`` is written from the thresholded + reverse-uniqueness
-constrained match sets, which are by construction a subset of the
candidates.
"""
from __future__ import annotations

import gc
import json
import logging
import time

import joblib
import pandas as pd

from .. import config
from ..blocking.candidate_generator import union_and_rank
from ..data import loader
from ..features.build_feature_matrix import build_feature_matrix
from ..models.one_to_one_constraint import enforce_reverse_uniqueness
from ..normalization.run_normalize import normalize_source
from ..postprocess.assemble_outputs import write_candidate_pairs, write_matching_results
from ..postprocess.sanity_checks import check_outputs
from .common import NORMALIZED_KEEP_COLS, load_normalize_block

logger = logging.getLogger(__name__)


def run_inference_pipeline(
    cfg: config.PipelineConfig = config.DEFAULT_CONFIG,
    nrows: int | None = None,
    matching_out=None,
    candidate_out=None,
) -> dict:
    t_start = time.time()
    report: dict = {}
    matching_out = matching_out or (config.OUTPUT_DIR / "matching_results.tsv")
    candidate_out = candidate_out or (config.OUTPUT_DIR / "candidate_pairs.tsv")

    logger.info("loading + normalizing S1 (nrows=%s)", nrows)
    s1 = loader.load_source(config.TEST_FILES["S1"], nrows=nrows)
    all_s1_ids = s1["entity_id"].tolist()
    s1n = normalize_source(s1)
    del s1
    s1n = s1n[NORMALIZED_KEEP_COLS].copy()
    gc.collect()

    logger.info("blocking (S2 and S3 processed sequentially to bound peak memory)")
    cand2, s2_subset = load_normalize_block(s1n, config.TEST_FILES["S2"], "S2", cfg.blocking, nrows=nrows)
    cand3, s3_subset = load_normalize_block(s1n, config.TEST_FILES["S3"], "S3", cfg.blocking, nrows=nrows)
    candidates = union_and_rank(pd.concat([cand2, cand3], ignore_index=True), cfg.blocking.max_candidates_per_s1)
    del cand2, cand3
    gc.collect()
    report["n_candidates"] = len(candidates)
    report["avg_candidates_per_s1"] = candidates.groupby("source1_entity_id").size().mean() if len(candidates) else 0.0

    logger.info("feature engineering")
    others_n = pd.concat([s2_subset, s3_subset], ignore_index=True)
    valid_cand_ids = set(others_n["entity_id"])
    del s2_subset, s3_subset
    gc.collect()
    X, meta = build_feature_matrix(candidates[["source1_entity_id", "candidate_entity_id", "score"]], s1n, others_n, cfg.features)

    logger.info("loading trained model + threshold")
    model = joblib.load(config.ARTIFACTS_DIR / "lgbm_model.joblib")
    feature_cols = json.loads((config.ARTIFACTS_DIR / "feature_columns.json").read_text())
    threshold = json.loads((config.ARTIFACTS_DIR / "threshold.json").read_text())["tau"]
    X = X.reindex(columns=feature_cols, fill_value=0)

    logger.info("scoring")
    proba = model.predict_proba(X)[:, 1]
    scored = meta.copy()
    scored["proba"] = proba

    logger.info("applying reverse-uniqueness constraint + threshold=%.3f", threshold)
    constrained = enforce_reverse_uniqueness(scored, "proba")
    kept = constrained[constrained["proba"] >= threshold]
    matches = kept.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()
    candidate_sets = candidates.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()

    matching_df = write_matching_results(all_s1_ids, matches, matching_out)
    candidate_df = write_candidate_pairs(all_s1_ids, candidate_sets, candidate_out)

    issues = check_outputs(matching_df, candidate_df, set(all_s1_ids), valid_cand_ids)
    report["sanity_issues"] = issues
    if issues:
        logger.warning("sanity issues found: %s", issues)
    else:
        logger.info("sanity checks passed")

    report["runtime_sec"] = time.time() - t_start
    logger.info("inference pipeline done in %.1fs", report["runtime_sec"])
    return report
