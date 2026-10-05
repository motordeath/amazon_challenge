"""Reverse uniqueness constraint: an S2/S3 record belongs to at most one S1.

This is NOT a 1-to-1 constraint on the S1 side (an S1 entity may still have
many matches). The EDA (section 7) found this holds for 100% of ground-truth
pairs: 0 of 3.69M matched S2 records and 0 of 3.94M matched S3 records map to
more than one S1 entity. If two different S1 entities both score a candidate
above threshold, at most one of them can be correct, so we keep only the
highest-scoring (S1, candidate) edge for each candidate and drop the rest —
a precision-only post-processing step (never adds a match, only removes
conflicting ones), which is exactly what a precision-weighted metric like
F_0.5 rewards.
"""
from __future__ import annotations

import pandas as pd


def enforce_reverse_uniqueness(scored: pd.DataFrame, score_col: str = "proba") -> pd.DataFrame:
    """Keep, for each candidate_entity_id, only its single highest-scoring S1 edge."""
    idx = scored.groupby("candidate_entity_id")[score_col].idxmax()
    return scored.loc[idx].reset_index(drop=True)
