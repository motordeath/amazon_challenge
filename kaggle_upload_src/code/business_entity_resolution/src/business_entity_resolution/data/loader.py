"""Memory-conscious TSV loading for the entity-resolution sources.

Source files use ``entity_id / business_name / business_address / country``.
Country is treated as an open-set string (never one-hot / hard-coded), per
the challenge spec: training only has US/India, test additionally has
France, and the pipeline must not assume a fixed vocabulary.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SOURCE_DTYPES = {
    "entity_id": "string[pyarrow]",
    "business_name": "string[pyarrow]",
    "business_address": "string[pyarrow]",
    "country": "category",
}

GT_DTYPES = {
    "source1_entity_id": "string[pyarrow]",
    "matched_entity_ids": "string[pyarrow]",
}


def load_source(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Load one source TSV (S1/S2/S3) with compact dtypes.

    ``business_address`` may contain the empty string / NaN (~3.3% of S2/S3);
    missingness is preserved rather than filled, so downstream feature code
    can build an explicit ``*_missing`` indicator instead of matching on a
    fabricated placeholder string.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=SOURCE_DTYPES,
        na_values=["", "NaN", "nan", "N/A"],
        keep_default_na=True,
        nrows=nrows,
    )
    expected_cols = {"entity_id", "business_name", "business_address", "country"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns {missing}")
    return df


def load_ground_truth(path: Path, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=GT_DTYPES,
        na_values=[],
        keep_default_na=False,
        nrows=nrows,
    )
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")
    return df


def explode_ground_truth(gt: pd.DataFrame) -> pd.DataFrame:
    """Long form: one row per (source1_entity_id, matched_entity_id) pair.

    Singletons (empty ``matched_entity_ids``) are dropped here since they
    have no positive pair to emit; the label-construction step re-adds them
    explicitly for decision-layer training/evaluation.
    """
    has_match = gt["matched_entity_ids"].str.len() > 0
    long = gt.loc[has_match, ["source1_entity_id", "matched_entity_ids"]].copy()
    long["matched_entity_id"] = long["matched_entity_ids"].str.split(",")
    long = long.explode("matched_entity_id", ignore_index=True)
    long = long.drop(columns=["matched_entity_ids"])
    long["matched_entity_id"] = long["matched_entity_id"].astype("string[pyarrow]")
    return long


def split_source_prefix(entity_id: pd.Series) -> pd.Series:
    """Return the source prefix (``S1``/``S2``/``S3``) of each entity id."""
    return entity_id.str.slice(0, 2)


def load_all_train(nrows: int | None = None) -> dict[str, pd.DataFrame]:
    from .. import config

    return {
        "S1": load_source(config.TRAIN_FILES["S1"], nrows=nrows),
        "S2": load_source(config.TRAIN_FILES["S2"], nrows=nrows),
        "S3": load_source(config.TRAIN_FILES["S3"], nrows=nrows),
        "ground_truth": load_ground_truth(config.TRAIN_GROUND_TRUTH, nrows=nrows),
    }


def load_all_test(nrows: int | None = None) -> dict[str, pd.DataFrame]:
    from .. import config

    return {
        "S1": load_source(config.TEST_FILES["S1"], nrows=nrows),
        "S2": load_source(config.TEST_FILES["S2"], nrows=nrows),
        "S3": load_source(config.TEST_FILES["S3"], nrows=nrows),
    }
