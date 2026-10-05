"""Label candidate pairs against ground truth (hard-negative-rich, per spec section 17).

Positives are candidate pairs that ARE a true match; negatives are every
other candidate pair that survived blocking — these are exactly the "hard
negatives" the task spec calls for (superficially similar enough to have
passed blocking, but wrong), rather than arbitrary random pairs.
"""
from __future__ import annotations

import pandas as pd


def label_candidates(candidates: pd.DataFrame, ground_truth_long: pd.DataFrame) -> pd.DataFrame:
    gt = ground_truth_long.rename(columns={"matched_entity_id": "candidate_entity_id"}).copy()
    gt["label"] = 1
    merged = candidates.merge(
        gt[["source1_entity_id", "candidate_entity_id", "label"]],
        on=["source1_entity_id", "candidate_entity_id"],
        how="left",
    )
    merged["label"] = merged["label"].fillna(0).astype("int8")
    return merged
