"""Exact / near-exact blocking channels: cheap, high-precision recall boosters.

These are plain equality joins (not fuzzy), so they never explode
combinatorially: an equi-join's output size is bounded by the actual key
collisions in the data, which the EDA shows are modest (duplicate business
names are legitimate distinct businesses, not degenerate hot keys).
"""
from __future__ import annotations

import pandas as pd


def exact_key_blocks(
    s1: pd.DataFrame,
    other: pd.DataFrame,
    key: str,
    channel_name: str,
) -> pd.DataFrame:
    """Join S1 to an S2/S3 frame on an exact-match key (e.g. name_compact).

    Both frames must carry ``entity_id`` and ``country``; the key column
    (e.g. ``name_compact``) must already be normalized. Empty-string keys are
    dropped (an empty normalized name/address must not become a universal
    join key that matches everything).
    """
    left = s1.loc[s1[key] != "", ["entity_id", "country", key]].rename(
        columns={"entity_id": "source1_entity_id"}
    )
    right = other.loc[other[key] != "", ["entity_id", "country", key]].rename(
        columns={"entity_id": "candidate_entity_id"}
    )
    merged = left.merge(right, on=["country", key], how="inner")
    merged["channel"] = channel_name
    return merged[["source1_entity_id", "candidate_entity_id", "channel"]]
