"""Name-similarity features computed row-wise over a candidate-pair frame.

Expects columns ``s1_name_normalized``, ``cand_name_normalized`` (and the
``_core``/``_tokens``/``_ngrams`` variants) already joined onto the pairs
frame by :mod:`build_feature_matrix`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein

from ._utils import any_overlap as _any_overlap
from ._utils import jaccard as _jaccard
from ._utils import length_ratio as _length_ratio
from ._utils import pairwise as _pairwise


def compute_name_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    feats["name_exact_normalized"] = (df["s1_name_normalized"] == df["cand_name_normalized"]).astype(np.int8)
    feats["name_exact_compact"] = (df["s1_name_compact"] == df["cand_name_compact"]).astype(np.int8)
    feats["name_exact_core"] = (df["s1_name_core"] == df["cand_name_core"]).astype(np.int8)

    feats["name_jaro_winkler"] = _pairwise(JaroWinkler.similarity, df["s1_name_normalized"], df["cand_name_normalized"])
    feats["name_levenshtein_ratio"] = _pairwise(Levenshtein.normalized_similarity, df["s1_name_normalized"], df["cand_name_normalized"])
    feats["name_core_jaro_winkler"] = _pairwise(JaroWinkler.similarity, df["s1_name_core"], df["cand_name_core"])

    s1_tokens = df["s1_name_normalized"].str.split(" ")
    cand_tokens = df["cand_name_normalized"].str.split(" ")
    feats["name_token_jaccard"] = _jaccard(s1_tokens, cand_tokens)
    feats["name_ngram_jaccard"] = _jaccard(df["s1_name_ngrams"], df["cand_name_ngrams"])
    feats["name_length_ratio"] = _length_ratio(df["s1_name_normalized"], df["cand_name_normalized"])

    if "s1_name_suffixes" in df.columns:
        feats["name_suffix_overlap"] = _any_overlap(df["s1_name_suffixes"], df["cand_name_suffixes"])
    else:
        feats["name_suffix_overlap"] = 0

    return feats
