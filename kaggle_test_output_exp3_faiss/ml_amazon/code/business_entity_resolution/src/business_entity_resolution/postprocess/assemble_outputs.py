"""Assemble the two required submission files from candidate/match dicts.

Exact format (task spec section "Output Format"):
    source1_entity_id<TAB>matched_entity_ids   (matching_results.tsv)
    source1_entity_id<TAB>candidate_entity_ids (candidate_pairs.tsv)
Comma-separated ID lists, no quoting, empty string for no matches/candidates,
exactly one row per required S1 entity.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _sets_to_frame(all_s1_ids: list[str], id_sets: dict[str, set], id_col: str, list_col: str) -> pd.DataFrame:
    rows = []
    for sid in all_s1_ids:
        ids = id_sets.get(sid, set())
        rows.append({id_col: sid, list_col: ",".join(sorted(ids))})
    return pd.DataFrame(rows)


def write_matching_results(all_s1_ids: list[str], matches: dict[str, set], path: Path) -> pd.DataFrame:
    df = _sets_to_frame(all_s1_ids, matches, "source1_entity_id", "matched_entity_ids")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return df


def write_candidate_pairs(all_s1_ids: list[str], candidates: dict[str, set], path: Path) -> pd.DataFrame:
    df = _sets_to_frame(all_s1_ids, candidates, "source1_entity_id", "candidate_entity_ids")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return df
