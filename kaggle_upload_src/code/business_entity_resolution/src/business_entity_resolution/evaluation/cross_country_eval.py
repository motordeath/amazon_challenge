"""Country-stratified F_0.5 breakdown, for error analysis / open-set checks."""
from __future__ import annotations

import pandas as pd

from .f_beta_macro import f_beta_score


def evaluate_by_country(
    predictions: dict[str, set],
    ground_truth: dict[str, set],
    s1_country: dict[str, str],
    beta: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for s1_id, actual in ground_truth.items():
        predicted = predictions.get(s1_id, set())
        rows.append({
            "source1_entity_id": s1_id,
            "country": s1_country.get(s1_id, "unknown"),
            "f_beta": f_beta_score(predicted, actual, beta=beta),
            "n_true": len(actual),
            "n_pred": len(predicted),
        })
    df = pd.DataFrame(rows)
    return df.groupby("country").agg(
        mean_f_beta=("f_beta", "mean"),
        n_entities=("f_beta", "size"),
        mean_true_matches=("n_true", "mean"),
        mean_pred_matches=("n_pred", "mean"),
    ).reset_index()
