"""Multi-channel candidate generation: union of exact + name-token + address-token blocking.

A candidate is retained if ANY channel retrieves it (union, never
intersection — intersecting blockers destroys recall, see task spec section
11F). Channels:

  A. exact name_compact match     (cheap, catches clean duplicates)
  B. name-token inverted index    (typo/abbreviation/word-order robust —
                                    see token.py)
  C. address-token inverted index (recovers Devanagari-name pairs via their
                                    still-Latin address, per EDA section 4)

All channels are partitioned by country (never cross-country — 100% of
positive pairs share country per EDA section 6).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import BlockingConfig
from .exact import exact_key_blocks
from .token import token_candidate_pairs

logger = logging.getLogger(__name__)


def generate_candidates_for_pair(
    s1: pd.DataFrame,
    other: pd.DataFrame,
    other_source_label: str,
    cfg: BlockingConfig,
) -> pd.DataFrame:
    """Candidates from one S1 frame against one of {S2, S3} (already normalized).

    Channels (union, never intersection — see task spec section 11F):
      A. exact name_compact match — cheap, catches clean duplicates.
      B. name-token inverted index — informative (low document-frequency)
         name tokens shared between S1 and the candidate; robust to
         word-order changes and single-token corruption since a multi-token
         name only needs ONE surviving clean token to be retrieved.
      C. address-token inverted index — recovers pairs where the name
         channel fails outright (e.g. Devanagari-script S2 names, per EDA
         section 4), since the address is still largely Latin-script.

    A brute-force TF-IDF char-ngram cosine kNN channel was tried first but
    measured at ~68ms/query against the full 3-5M-doc corpus (~25h for one
    channel over the full S1) — infeasible at this scale. The inverted-index
    hash-join here processes the full US partition (1.32M S1 x 3.0M S2) in
    ~52s because it only ever computes a score for pairs that already share
    a token, instead of scoring every corpus document against every query.
    """
    channels = []

    channels.append(exact_key_blocks(s1, other, "name_compact", f"exact_name_{other_source_label}"))

    channels.append(
        token_candidate_pairs(
            s1, other, "name_normalized", f"tok_name_{other_source_label}",
            min_token_length=cfg.min_token_length,
            max_df_fraction=cfg.max_token_doc_frequency,
            max_df_abs_other=cfg.token_max_df_abs_other,
            max_df_abs_s1=cfg.token_max_df_abs_s1,
            top_k_per_s1=cfg.token_top_k_per_s1,
        )
    )
    channels.append(
        token_candidate_pairs(
            s1, other, "address_normalized", f"tok_address_{other_source_label}",
            min_token_length=cfg.min_token_length,
            max_df_fraction=cfg.max_token_doc_frequency,
            max_df_abs_other=cfg.token_max_df_abs_other,
            max_df_abs_s1=cfg.token_max_df_abs_s1,
            top_k_per_s1=cfg.token_top_k_per_s1,
        )
    )

    combined = pd.concat(channels, ignore_index=True)
    return combined


def union_and_rank(candidates: pd.DataFrame, max_per_s1: int) -> pd.DataFrame:
    """Union channels per (source1, candidate) pair, keep best score, cap top-K per S1.

    ``score`` is NaN for exact-match channels (no fuzzy score computed); those
    are treated as maximal confidence (1.0) since an equality join is exact.
    """
    df = candidates.copy()
    df["score"] = df["score"].fillna(1.0)
    df["is_exact"] = df["channel"].str.startswith("exact_")

    agg = (
        df.groupby(["source1_entity_id", "candidate_entity_id"], as_index=False)
        .agg(score=("score", "max"), is_exact=("is_exact", "max"))
    )
    # exact matches always float to the top regardless of fuzzy score.
    agg["rank_score"] = agg["score"] + agg["is_exact"].astype(int) * 10.0
    agg["rank"] = agg.groupby("source1_entity_id")["rank_score"].rank(method="first", ascending=False)
    kept = agg[agg["rank"] <= max_per_s1].drop(columns=["rank_score", "rank", "is_exact"])
    return kept
