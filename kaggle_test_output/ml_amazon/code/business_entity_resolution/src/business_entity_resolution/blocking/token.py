"""Inverted-index token blocking via hash-join, not brute-force similarity search.

Brute-force cosine kNN (TF-IDF against the full 3-5M-document corpus) is too
slow at this data scale (~68ms/query measured, i.e. >24h for one channel over
the full S1) because it computes a similarity for every corpus document on
every query, even documents that share no token with the query. An inverted
index avoids this: two records can only ever become candidates if they share
at least one sufficiently rare token, so the cost is bounded by actual token
co-occurrence (a hash join) rather than by corpus size.

Non-discriminative tokens (``inc``, generic words appearing in >``max_df``
of a partition) are dropped from the index entirely rather than down-weighted,
since the whole point is to keep each posting list short.
"""
from __future__ import annotations

import gc
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _explode(df: pd.DataFrame, text_col: str, min_token_length: int) -> pd.DataFrame:
    """Tokenize on the fly by splitting the normalized string column.

    Tokens are never stored as a persistent list column (see
    ``run_normalize``): ``name_tokens`` is just ``name_normalized.split(" ")``,
    so storing both would double the object overhead for the same
    information across the full corpus. The split happens here instead,
    scoped to whatever (already country-filtered, 2-column) slice is passed
    in, so the exploded intermediate never outlives this call.
    """
    tokens = df[text_col].str.split(" ")
    long = pd.DataFrame({"entity_id": df["entity_id"].to_numpy(), "token": tokens.to_numpy()})
    long = long.explode("token", ignore_index=True)
    long = long[long["token"].notna() & (long["token"].str.len() >= min_token_length)]
    return long


def _cap_by_doc_frequency(long: pd.DataFrame, max_df_fraction: float, max_df_abs: int) -> pd.DataFrame:
    """Drop tokens whose document frequency (within this already country-scoped frame) exceeds the cap."""
    part_size = long["entity_id"].nunique()
    token_count = long.groupby("token")["entity_id"].transform("nunique")
    cap = min(part_size * max_df_fraction, max_df_abs)
    return long[token_count <= cap]


def _token_candidates_one_country(
    s1_c: pd.DataFrame,
    other_c: pd.DataFrame,
    text_col: str,
    min_token_length: int,
    max_df_fraction: float,
    max_df_abs_other: int,
    max_df_abs_s1: int,
    top_k_per_s1: int,
) -> pd.DataFrame:
    s1_long = _cap_by_doc_frequency(_explode(s1_c, text_col, min_token_length), max_df_fraction, max_df_abs_s1)
    other_long = _cap_by_doc_frequency(_explode(other_c, text_col, min_token_length), max_df_fraction, max_df_abs_other)
    s1_long = s1_long.rename(columns={"entity_id": "source1_entity_id"})
    other_long = other_long.rename(columns={"entity_id": "candidate_entity_id"})

    merged = s1_long.merge(other_long, on="token", how="inner")
    del s1_long, other_long
    if merged.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])

    scored = (
        merged.groupby(["source1_entity_id", "candidate_entity_id"], as_index=False)
        .agg(shared_tokens=("token", "size"))
    )
    del merged
    max_shared = scored["shared_tokens"].max()
    scored["score"] = scored["shared_tokens"] / max(max_shared, 1)

    rank = scored.groupby("source1_entity_id")["shared_tokens"].rank(method="first", ascending=False)
    scored = scored[rank <= top_k_per_s1]
    return scored[["source1_entity_id", "candidate_entity_id", "score"]]


def token_candidate_pairs(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    text_col: str,
    channel_name: str,
    min_token_length: int = 2,
    max_df_fraction: float = 0.001,
    max_df_abs_other: int = 400,
    max_df_abs_s1: int = 2000,
    top_k_per_s1: int = 40,
) -> pd.DataFrame:
    """Candidate pairs sharing at least one informative token, scored by shared-token count.

    Processes one country partition at a time (never a single explode/merge
    across the whole multi-country frame) so peak memory is bounded by the
    largest single country partition, not the full corpus.

    ``top_k_per_s1`` caps output size *per channel*, before the multi-channel
    union: without it, a single channel over the full corpus can return tens
    of millions of (s1, candidate) pairs (an S1 name with several
    moderately-common tokens can co-occur with dozens of candidates each),
    which then multiplies across name/address x S2/S3 x country before the
    final ``union_and_rank`` cap ever gets a chance to run — the dominant
    cause of the OOM hit during initial full-scale testing.
    """
    countries = sorted(set(s1_df["country"].dropna().astype(str)) & set(other_df["country"].dropna().astype(str)))
    results = []
    for country in countries:
        s1_c = s1_df.loc[s1_df["country"] == country, ["entity_id", text_col]]
        other_c = other_df.loc[other_df["country"] == country, ["entity_id", text_col]]
        if s1_c.empty or other_c.empty:
            continue
        part = _token_candidates_one_country(
            s1_c, other_c, text_col, min_token_length, max_df_fraction,
            max_df_abs_other, max_df_abs_s1, top_k_per_s1,
        )
        results.append(part)
        del s1_c, other_c, part
        gc.collect()

    if not results:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "channel", "score"])
    out = pd.concat(results, ignore_index=True)
    out["channel"] = channel_name
    return out[["source1_entity_id", "candidate_entity_id", "channel", "score"]]
