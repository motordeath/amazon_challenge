"""Blocking recall: the ceiling on everything downstream.

A true match that never becomes a candidate can never be recovered by the
classifier, so this is the single most important blocking metric.

Uses a merge-based (hash-join) computation rather than building a Python
``set`` of (s1, candidate) tuples: at full scale ``candidates`` can have on
the order of 10^8 rows, and constructing a Python set of that many tuples is
both slow (~minutes) and memory-heavy (Python tuple/object overhead per
entry) compared to a vectorized pandas merge with an indicator column.
"""
from __future__ import annotations

import pandas as pd


def blocking_recall(
    candidates: pd.DataFrame,
    ground_truth_long: pd.DataFrame,
    breakdown_cols: dict[str, pd.Series] | None = None,
) -> dict:
    """Compute overall + optional breakdown blocking recall.

    ``candidates`` has columns [source1_entity_id, candidate_entity_id].
    ``ground_truth_long`` has columns [source1_entity_id, matched_entity_id]
    (one row per true positive pair; from ``loader.explode_ground_truth``).
    """
    cand_keys = candidates[["source1_entity_id", "candidate_entity_id"]].drop_duplicates()
    gt = ground_truth_long.merge(
        cand_keys,
        left_on=["source1_entity_id", "matched_entity_id"],
        right_on=["source1_entity_id", "candidate_entity_id"],
        how="left",
        indicator=True,
    )
    gt["retrieved"] = gt["_merge"] == "both"
    gt = gt.drop(columns=["_merge", "candidate_entity_id"])

    report = {
        "total_true_pairs": len(gt),
        "retrieved_true_pairs": int(gt["retrieved"].sum()),
        "blocking_recall": float(gt["retrieved"].mean()) if len(gt) else float("nan"),
        "n_candidate_pairs": len(candidates),
        "avg_candidates_per_s1": candidates.groupby("source1_entity_id").size().mean() if len(candidates) else 0.0,
    }

    if breakdown_cols:
        report["breakdown"] = {}
        for name, series in breakdown_cols.items():
            gt_b = gt.copy()
            gt_b["_grp"] = gt_b["source1_entity_id"].map(series)
            by_group = gt_b.groupby("_grp")["retrieved"].mean()
            report["breakdown"][name] = by_group.to_dict()

    return report


def find_missed_matches(candidates: pd.DataFrame, ground_truth_long: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``ground_truth_long`` never retrieved by blocking — for error analysis."""
    cand_keys = candidates[["source1_entity_id", "candidate_entity_id"]].drop_duplicates()
    merged = ground_truth_long.merge(
        cand_keys,
        left_on=["source1_entity_id", "matched_entity_id"],
        right_on=["source1_entity_id", "candidate_entity_id"],
        how="left",
        indicator=True,
    )
    missed = merged[merged["_merge"] == "left_only"]
    return missed[["source1_entity_id", "matched_entity_id"]].reset_index(drop=True)
