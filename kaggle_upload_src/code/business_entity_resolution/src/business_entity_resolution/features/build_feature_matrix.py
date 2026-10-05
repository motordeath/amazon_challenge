"""Assemble the full feature matrix for a candidate-pairs frame.

Joins S1 attributes and S2/S3 candidate attributes onto each pair, then
computes the name/address/country/rank feature families and concatenates
them into one numeric matrix ready for the LightGBM ranker.
"""
from __future__ import annotations

import gc
import logging

import pandas as pd

from ..config import FeatureConfig
from ..normalization.run_normalize import add_ngram_columns
from .address_similarity import compute_address_features
from .country_features import compute_country_features
from .name_similarity import compute_name_features
from .rank_features import compute_rank_features

logger = logging.getLogger(__name__)

# Feature computation joins every S1/candidate string+list attribute (incl.
# character-ngram lists) onto each candidate-pair row. Materializing that
# joined frame for the full candidate set at once was the dominant memory
# cost in full-scale runs — far larger than the eventual numeric feature
# matrix, since ngram lists get physically duplicated per row. Processing in
# chunks bounds peak memory to one chunk's joined frame; only the small
# numeric feature chunks are retained across iterations.
CHUNK_SIZE = 4_000_000

S1_COLS = [
    "entity_id", "country", "name_normalized", "name_compact", "name_core",
    "name_ngrams", "name_suffixes", "name_missing",
    "address_normalized", "address_compact", "address_numbers",
    "address_ngrams", "address_missing",
]
CAND_COLS = S1_COLS  # same schema for S2/S3


def _prep_lookup(df: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
    sub = df[cols].copy()
    sub = sub.rename(columns={c: (c if c == "entity_id" else f"{prefix}{c}") for c in cols})
    return sub


def _compute_chunk(
    chunk: pd.DataFrame,
    s1_lookup: pd.DataFrame,
    cand_lookup: pd.DataFrame,
    cfg: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = chunk.merge(s1_lookup, on="source1_entity_id", how="left")
    joined = joined.merge(cand_lookup, on="candidate_entity_id", how="left")

    feature_frames = []
    if cfg.use_name_features:
        feature_frames.append(compute_name_features(joined))
    if cfg.use_address_features:
        feature_frames.append(compute_address_features(joined))
    if cfg.use_cross_features:
        feature_frames.append(compute_country_features(joined))
    if cfg.use_rank_features:
        feature_frames.append(compute_rank_features(joined))

    X_chunk = pd.concat(feature_frames, axis=1).reset_index(drop=True)
    meta_chunk = joined[["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)
    return X_chunk, meta_chunk


def build_feature_matrix(
    candidates: pd.DataFrame,
    s1_normalized: pd.DataFrame,
    others_normalized: pd.DataFrame,
    cfg: FeatureConfig,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (X, meta) where meta = candidates[[source1_entity_id, candidate_entity_id]].

    N-grams are added here, on the frames already reduced to just the S1 /
    candidate entities referenced by ``candidates`` — never on a full source
    corpus (see ``normalization.run_normalize``). The join against those
    lookups (and the resulting feature computation) is then done in chunks
    over ``candidates`` — see module docstring for why.
    """
    # .unique() first: candidates can have tens of millions of rows but only
    # up to ~2.2M/~6M distinct S1/candidate ids — isin() against the raw
    # column would hash every row instead of every distinct id.
    s1_used = s1_normalized[s1_normalized["entity_id"].isin(candidates["source1_entity_id"].unique())]
    cand_used = others_normalized[others_normalized["entity_id"].isin(candidates["candidate_entity_id"].unique())]
    s1_used = add_ngram_columns(s1_used)
    cand_used = add_ngram_columns(cand_used)

    s1_lookup = _prep_lookup(s1_used, S1_COLS, "s1_").rename(columns={"entity_id": "source1_entity_id"})
    cand_lookup = _prep_lookup(cand_used, CAND_COLS, "cand_").rename(columns={"entity_id": "candidate_entity_id"})
    del s1_used, cand_used

    n = len(candidates)
    if n <= chunk_size:
        X, meta = _compute_chunk(candidates, s1_lookup, cand_lookup, cfg)
        return X, meta

    logger.info("feature engineering in %d chunks of <=%d rows", (n + chunk_size - 1) // chunk_size, chunk_size)
    x_chunks, meta_chunks = [], []
    for start in range(0, n, chunk_size):
        chunk = candidates.iloc[start : start + chunk_size]
        x_chunk, meta_chunk = _compute_chunk(chunk, s1_lookup, cand_lookup, cfg)
        x_chunks.append(x_chunk)
        meta_chunks.append(meta_chunk)
        del chunk, x_chunk, meta_chunk
        gc.collect()

    X = pd.concat(x_chunks, ignore_index=True)
    meta = pd.concat(meta_chunks, ignore_index=True)
    return X, meta
