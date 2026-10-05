"""Candidate-pool structural features: blocking score/rank, pool size.

These let the classifier reason about a candidate relative to its peers for
the same S1 entity (e.g. "this is the best of 3 candidates" vs "one of 40
weak candidates"), which the multi-match decision layer also consumes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    feats["blocking_score"] = df["score"].astype(np.float32)
    pool_size = df.groupby("source1_entity_id")["candidate_entity_id"].transform("size")
    feats["candidate_pool_size"] = pool_size.astype(np.int16)

    rank = df.groupby("source1_entity_id")["score"].rank(method="first", ascending=False)
    feats["blocking_rank"] = rank.astype(np.int16)
    best_score = df.groupby("source1_entity_id")["score"].transform("max")
    feats["score_gap_to_best"] = (best_score - df["score"]).astype(np.float32)

    src_from_s2 = df["candidate_entity_id"].str.startswith("S2-").astype(np.int8)
    feats["candidate_is_s2"] = src_from_s2
    return feats
