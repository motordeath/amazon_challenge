"""Score a candidate-pairs feature matrix with a trained model."""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd


def score_candidates(model: lgb.LGBMClassifier, X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(X)[:, 1]
    out = meta.copy()
    out["proba"] = proba
    return out
