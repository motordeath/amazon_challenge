"""Input-data validation: schema, ID-prefix integrity, missingness."""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_source(df: pd.DataFrame, expected_prefix: str, name: str) -> dict:
    report: dict = {"name": name, "rows": len(df)}

    bad_prefix = ~df["entity_id"].str.startswith(f"{expected_prefix}-")
    report["bad_prefix_count"] = int(bad_prefix.sum())
    if report["bad_prefix_count"]:
        logger.warning("%s: %d entity_ids missing prefix %s-", name, report["bad_prefix_count"], expected_prefix)

    dup = df["entity_id"].duplicated()
    report["duplicate_id_count"] = int(dup.sum())

    report["name_missing"] = int(df["business_name"].isna().sum())
    report["address_missing"] = int(df["business_address"].isna().sum())
    report["country_missing"] = int(df["country"].isna().sum())
    both_missing = df["business_name"].isna() & df["business_address"].isna()
    report["both_missing"] = int(both_missing.sum())

    report["countries"] = sorted(df["country"].dropna().astype(str).unique().tolist())
    return report


def validate_ground_truth(gt: pd.DataFrame, s1: pd.DataFrame) -> dict:
    report: dict = {"rows": len(gt)}
    report["s1_coverage"] = int(gt["source1_entity_id"].isin(s1["entity_id"]).sum())
    report["missing_s1_coverage"] = len(gt) - report["s1_coverage"]

    n_matches = gt["matched_entity_ids"].str.len().gt(0)
    counts = gt["matched_entity_ids"].where(n_matches, "").str.split(",").apply(
        lambda x: 0 if x is None or x == [""] else len(x)
    )
    report["singleton_count"] = int((counts == 0).sum())
    report["singleton_rate"] = float((counts == 0).mean())
    report["mean_matches"] = float(counts.mean())
    report["max_matches"] = int(counts.max())
    return report


def log_report(report: dict) -> None:
    for k, v in report.items():
        logger.info("  %s: %s", k, v)
