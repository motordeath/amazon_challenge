"""Shared row-wise similarity helpers used by name_similarity and address_similarity."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pairwise(func, a: pd.Series, b: pd.Series) -> np.ndarray:
    # float32: at full scale the feature matrix has ~15 similarity columns
    # over tens of millions of candidate pairs, where float64 vs float32
    # is the difference between single- and double-digit GB.
    return np.fromiter((func(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a))


def jaccard(a: pd.Series, b: pd.Series) -> np.ndarray:
    out = np.empty(len(a), dtype=np.float32)
    for i, (sa, sb) in enumerate(zip(a, b)):
        sa, sb = set(sa), set(sb)
        if not sa and not sb:
            out[i] = 1.0
        elif not sa or not sb:
            out[i] = 0.0
        else:
            out[i] = len(sa & sb) / len(sa | sb)
    return out


def any_overlap(a: pd.Series, b: pd.Series) -> np.ndarray:
    """1 if the two sets share at least one element, else 0 (empty-empty -> 0, not 1)."""
    out = np.empty(len(a), dtype=np.int8)
    for i, (sa, sb) in enumerate(zip(a, b)):
        out[i] = 1 if (set(sa) & set(sb)) else 0
    return out


def length_ratio(a: pd.Series, b: pd.Series) -> np.ndarray:
    la = a.str.len().to_numpy(dtype=np.float32)
    lb = b.str.len().to_numpy(dtype=np.float32)
    denom = np.maximum(np.maximum(la, lb), 1.0)
    return 1.0 - np.abs(la - lb) / denom
