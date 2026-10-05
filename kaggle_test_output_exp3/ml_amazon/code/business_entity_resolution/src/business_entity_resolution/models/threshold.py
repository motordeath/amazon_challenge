"""Precision-first decision layer: turn per-candidate scores into match sets.

Two decision strategies are implemented and compared empirically on
validation data (never hard-coded as "obviously right" — see task spec
section 21):

* ``global_threshold``: keep every candidate with proba >= tau.
* ``margin``: keep candidates with proba >= tau AND proba >= best_proba -
  margin (relative-confidence filtering; helps reject a lone mediocre
  candidate while still keeping a tight cluster of near-tied true matches).

Singletons naturally fall out of both: if no candidate clears the bar, the
S1 entity gets an empty prediction, which the F_0.5 metric rewards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ThresholdConfig
from ..evaluation.f_beta_macro import macro_f_beta
from .one_to_one_constraint import enforce_reverse_uniqueness


def _to_pred_dict(kept: pd.DataFrame) -> dict[str, set]:
    return kept.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()


def apply_global_threshold(scored: pd.DataFrame, tau: float, score_col: str = "proba") -> pd.DataFrame:
    return scored[scored[score_col] >= tau]


def apply_margin_rule(scored: pd.DataFrame, tau: float, margin: float, score_col: str = "proba") -> pd.DataFrame:
    best = scored.groupby("source1_entity_id")[score_col].transform("max")
    return scored[(scored[score_col] >= tau) & (scored[score_col] >= best - margin)]


def tune_global_threshold(
    scored_val: pd.DataFrame,
    ground_truth: dict[str, set],
    all_s1_ids: list[str],
    cfg: ThresholdConfig,
    apply_reverse_constraint: bool = True,
    score_col: str = "proba",
) -> dict:
    """Grid search tau to maximize macro F_0.5, evaluated over ALL S1 ids
    (including ones with zero surviving candidates -> empty prediction)."""
    base = enforce_reverse_uniqueness(scored_val, score_col) if apply_reverse_constraint else scored_val

    results = []
    best = {"tau": None, "macro_f_beta": -1.0}
    for tau in np.arange(cfg.grid_min, cfg.grid_max + 1e-9, cfg.grid_step):
        kept = base[base[score_col] >= tau]
        preds = _to_pred_dict(kept)
        preds = {sid: preds.get(sid, set()) for sid in all_s1_ids}
        score = macro_f_beta(preds, ground_truth, beta=cfg.beta)
        results.append({"tau": round(float(tau), 4), "macro_f_beta": score["macro_f_beta"]})
        if score["macro_f_beta"] > best["macro_f_beta"]:
            best = {"tau": float(tau), "macro_f_beta": score["macro_f_beta"]}

    return {"best": best, "grid": pd.DataFrame(results)}


def predict_matches(
    scored: pd.DataFrame,
    tau: float,
    apply_reverse_constraint: bool = True,
    score_col: str = "proba",
) -> dict[str, set]:
    base = enforce_reverse_uniqueness(scored, score_col) if apply_reverse_constraint else scored
    kept = base[base[score_col] >= tau]
    return _to_pred_dict(kept)
