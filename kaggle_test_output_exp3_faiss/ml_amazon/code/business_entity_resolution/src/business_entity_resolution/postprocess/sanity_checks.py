"""Pre-submission invariant checks (in addition to the official validator)."""
from __future__ import annotations

import pandas as pd


def check_outputs(
    matching_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    required_s1_ids: set[str],
    valid_cand_ids: set[str],
) -> list[str]:
    issues = []

    if set(matching_df["source1_entity_id"]) != required_s1_ids:
        missing = required_s1_ids - set(matching_df["source1_entity_id"])
        extra = set(matching_df["source1_entity_id"]) - required_s1_ids
        if missing:
            issues.append(f"matching_results missing {len(missing)} required S1 ids")
        if extra:
            issues.append(f"matching_results has {len(extra)} unexpected S1 ids")

    if matching_df["source1_entity_id"].duplicated().any():
        issues.append("duplicate source1_entity_id rows in matching_results")

    for _, row in matching_df.iterrows():
        ids = [i for i in str(row["matched_entity_ids"]).split(",") if i]
        if len(ids) != len(set(ids)):
            issues.append(f"duplicate ids within match list for {row['source1_entity_id']}")
            break
        bad = [i for i in ids if not (i.startswith("S2-") or i.startswith("S3-")) or i not in valid_cand_ids]
        if bad:
            issues.append(f"invalid candidate ids in match list for {row['source1_entity_id']}: {bad[:3]}")
            break

    match_map = {
        r["source1_entity_id"]: set(str(r["matched_entity_ids"]).split(",")) - {""}
        for _, r in matching_df.iterrows()
    }
    cand_map = {
        r["source1_entity_id"]: set(str(r["candidate_entity_ids"]).split(",")) - {""}
        for _, r in candidate_df.iterrows()
    }
    for sid, matched in match_map.items():
        if not matched.issubset(cand_map.get(sid, set())):
            issues.append(f"{sid}: matched ids not a subset of candidate ids")
            break

    return issues
