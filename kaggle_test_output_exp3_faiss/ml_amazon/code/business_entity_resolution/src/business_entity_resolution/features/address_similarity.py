"""Address-similarity features computed row-wise over a candidate-pair frame."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein

from ._utils import jaccard as _jaccard
from ._utils import length_ratio as _length_ratio
from ._utils import pairwise as _pairwise


def compute_address_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    s1_missing = df["s1_address_missing"].to_numpy()
    cand_missing = df["cand_address_missing"].to_numpy()
    feats["s1_address_missing"] = s1_missing.astype(np.int8)
    feats["cand_address_missing"] = cand_missing.astype(np.int8)
    feats["either_address_missing"] = (s1_missing | cand_missing).astype(np.int8)

    feats["address_exact_normalized"] = (df["s1_address_normalized"] == df["cand_address_normalized"]).astype(np.int8)
    feats["address_exact_compact"] = (df["s1_address_compact"] == df["cand_address_compact"]).astype(np.int8)

    feats["address_jaro_winkler"] = _pairwise(JaroWinkler.similarity, df["s1_address_normalized"], df["cand_address_normalized"])
    feats["address_levenshtein_ratio"] = _pairwise(Levenshtein.normalized_similarity, df["s1_address_normalized"], df["cand_address_normalized"])
    s1_tokens = df["s1_address_normalized"].str.split(" ")
    cand_tokens = df["cand_address_normalized"].str.split(" ")
    feats["address_token_jaccard"] = _jaccard(s1_tokens, cand_tokens)
    feats["address_ngram_jaccard"] = _jaccard(df["s1_address_ngrams"], df["cand_address_ngrams"])
    feats["address_number_jaccard"] = _jaccard(df["s1_address_numbers"], df["cand_address_numbers"])
    feats["address_length_ratio"] = _length_ratio(df["s1_address_normalized"], df["cand_address_normalized"])

    # neutralize similarity scores when either side is missing, so the
    # missingness indicators (not a fabricated "0 similarity") carry the signal.
    missing_mask = (s1_missing | cand_missing)
    for col in ["address_jaro_winkler", "address_levenshtein_ratio", "address_token_jaccard", "address_ngram_jaccard"]:
        feats.loc[missing_mask, col] = np.nan

    return feats
