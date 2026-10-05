"""Stratified S1 subsampling for a fast training/validation mode.

Used only to shrink the *S1 query side* of the training pipeline (e.g. for
faster iteration on memory-constrained hardware) — S2/S3 are never
subsampled here, so blocking still runs against the full, realistic
candidate corpus and blocking recall remains meaningful. This is a
training-time convenience only: ``inference_pipeline`` never calls this and
always scores every S1 entity in the test set.

Stratifies on (country, match_count) so the sample preserves both the
country mix and the ground-truth match-cardinality distribution — including
the singleton (match_count == 0) stratum — as closely as proportional
sampling allows.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _match_counts(gt: pd.DataFrame) -> pd.DataFrame:
    """Returns a frame [source1_entity_id, match_count] (0 for singletons)."""
    has_match = gt["matched_entity_ids"].str.len() > 0
    n = np.where(has_match, gt["matched_entity_ids"].str.count(",") + 1, 0)
    return pd.DataFrame({"entity_id": gt["source1_entity_id"].to_numpy(), "match_count": n.astype("int16")})


def stratified_sample_s1(
    s1_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    n_samples: int,
    seed: int = 42,
) -> pd.Index:
    """Returns a pandas Index of sampled ``entity_id`` values.

    Stratum = (country, match_count). Each stratum is sampled at the same
    fraction (``n_samples / len(s1_df)``), so both the country mix and the
    0/1/2/.../11-match cardinality distribution from the EDA are preserved
    proportionally, not just approximated by luck by unstratified random
    sampling.
    """
    if n_samples >= len(s1_df):
        return s1_df["entity_id"]

    counts = _match_counts(gt_df)
    strata = s1_df[["entity_id", "country"]].merge(counts, on="entity_id", how="left")
    strata["match_count"] = strata["match_count"].fillna(0).astype("int16")

    frac = n_samples / len(s1_df)
    sampled = (
        strata.groupby(["country", "match_count"], group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=seed))
    )
    logger.info(
        "stratified S1 sample: requested=%d actual=%d (%.2f%% of %d), strata=%d",
        n_samples, len(sampled), 100 * len(sampled) / len(s1_df), len(s1_df),
        strata.groupby(["country", "match_count"]).ngroups,
    )
    return pd.Index(sampled["entity_id"])


def sample_report(s1_df: pd.DataFrame, gt_df: pd.DataFrame, sampled_ids: pd.Index) -> dict:
    """Diagnostic comparison of full-population vs. sampled-population distributions."""
    counts = _match_counts(gt_df)
    full = s1_df[["entity_id", "country"]].merge(counts, on="entity_id", how="left")
    full["match_count"] = full["match_count"].fillna(0).astype("int16")
    sampled = full[full["entity_id"].isin(sampled_ids)]

    def dist(df: pd.DataFrame) -> dict:
        return {
            "n": len(df),
            "country_pct": (df["country"].value_counts(normalize=True) * 100).round(2).to_dict(),
            "singleton_pct": round(100 * (df["match_count"] == 0).mean(), 3),
            "match_count_pct": (df["match_count"].value_counts(normalize=True).sort_index() * 100).round(2).to_dict(),
        }

    return {"full_population": dist(full), "sample": dist(sampled)}
