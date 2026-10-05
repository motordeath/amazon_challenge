"""Official evaluation metric: macro-averaged F_beta (beta=0.5) over S1 entities.

F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)

Computed per S1 entity then macro-averaged across ALL S1 entities in the
evaluation set, including singletons: a correctly-predicted empty list on a
true singleton scores 1.0; any predicted match on a true singleton scores 0.0.
"""
from __future__ import annotations

import pandas as pd


def f_beta_score(predicted: set, actual: set, beta: float = 0.5) -> float:
    if not actual and not predicted:
        return 1.0
    if not predicted or not actual:
        return 0.0
    tp = len(predicted & actual)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(actual)
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def macro_f_beta(
    predictions: dict[str, set],
    ground_truth: dict[str, set],
    beta: float = 0.5,
) -> dict:
    """predictions/ground_truth: source1_entity_id -> set of matched entity_ids.

    Every key in ``ground_truth`` must have an entry in ``predictions``
    (missing entities are treated as an empty prediction, matching the
    competition's "every S1 must appear" requirement — but note a truly
    missing row would be a submission-format violation, not merely a score
    of 0, in the real validator).
    """
    scores = []
    for s1_id, actual in ground_truth.items():
        predicted = predictions.get(s1_id, set())
        scores.append(f_beta_score(predicted, actual, beta=beta))
    return {
        "macro_f_beta": sum(scores) / len(scores) if scores else float("nan"),
        "n_entities": len(scores),
    }


def macro_f_beta_from_frames(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    beta: float = 0.5,
    pred_col: str = "matched_entity_ids",
    gt_col: str = "matched_entity_ids",
) -> dict:
    """Both frames: [source1_entity_id, <col> as comma-separated string]."""

    def to_dict(df: pd.DataFrame, col: str) -> dict[str, set]:
        out = {}
        for row in df.itertuples(index=False):
            sid = getattr(row, "source1_entity_id")
            raw = getattr(row, col)
            out[sid] = set(raw.split(",")) if raw else set()
        return out

    return macro_f_beta(to_dict(pred_df, pred_col), to_dict(gt_df, gt_col), beta=beta)
