"""Shared memory-frugal load -> normalize -> block step, reused by train/inference.

Loads and fully normalizes one S2/S3 file, blocks it against the (already
normalized, kept in memory) S1 frame, then immediately reduces it to just the
rows referenced by the resulting candidates before discarding the full
normalized frame. This keeps peak memory bounded by roughly one full S2/S3
normalized frame at a time instead of holding S2 and S3 simultaneously.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import pandas as pd

from ..blocking.candidate_generator import generate_candidates_for_pair
from ..config import BlockingConfig
from ..data import loader
from ..normalization.run_normalize import normalize_source

logger = logging.getLogger(__name__)

# Columns actually needed downstream (feature engineering re-adds n-grams on
# demand — see build_feature_matrix). Raw business_name/business_address are
# dropped here: they're only useful for debugging, and keeping both the raw
# and every derived string column for every referenced entity was a real
# memory cost when this subset is held across the union_and_rank/labeling
# stage (a full-scale run measured this as part of a peak that triggered an
# OOM kill).
NORMALIZED_KEEP_COLS = [
    "entity_id", "country", "name_normalized", "name_compact", "name_core",
    "name_suffixes", "name_missing", "address_normalized", "address_compact",
    "address_numbers", "address_missing",
]


def load_normalize_block(
    s1n: pd.DataFrame,
    source_path: Path,
    source_label: str,
    cfg: BlockingConfig,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (candidates, reduced_normalized_subset) for one of {S2, S3}."""
    raw = loader.load_source(source_path, nrows=nrows)
    normalized = normalize_source(raw)
    del raw
    gc.collect()

    candidates = generate_candidates_for_pair(s1n, normalized, source_label, cfg)

    used_ids = set(candidates["candidate_entity_id"].unique())
    subset = normalized.loc[normalized["entity_id"].isin(used_ids), NORMALIZED_KEEP_COLS].copy()
    del normalized
    gc.collect()

    logger.info(
        "%s: %d candidate rows referencing %d unique entities (kept in memory for features)",
        source_label, len(candidates), len(subset),
    )
    return candidates, subset
