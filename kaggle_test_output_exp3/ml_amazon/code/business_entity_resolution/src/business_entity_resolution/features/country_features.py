"""Cross-field country features.

Country is an open-set string label (US/India in training, +France in test),
so this only ever compares equality — never one-hot encodes or hard-codes a
vocabulary. Blocking is already country-strict, so this is expected to be
1.0 for nearly all candidates; it is kept as a defensive feature in case a
future blocking channel relaxes that constraint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_country_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    feats["country_match"] = (df["s1_country"].astype(str) == df["cand_country"].astype(str)).astype(np.int8)
    return feats
