"""Character n-gram TF-IDF blocking channel.

For each country partition (blocking is always country-strict — see EDA
section 6/13: 100% of positive pairs share country, so cross-country pairs
are pure noise), fits a char n-gram TF-IDF vectorizer on the S2+S3 corpus and
retrieves the top-k nearest S2/S3 candidates per S1 record by cosine
similarity. Char n-grams (not word tokens) make this robust to typos,
abbreviations and word-order changes, and it is the primary channel for
catching non-Latin-script name pairs' *address* similarity when run on the
address column instead of the name column.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def _tfidf_knn_for_country(
    s1_ids: np.ndarray,
    s1_text: np.ndarray,
    other_ids: np.ndarray,
    other_text: np.ndarray,
    k: int,
    ngram_range: tuple[int, int],
    max_features: int,
    min_similarity: float,
    query_batch_size: int,
    max_df: float = 0.2,
    min_df: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
        dtype=np.float32,
    )
    other_matrix = vectorizer.fit_transform(other_text)
    s1_matrix = vectorizer.transform(s1_text)

    n_neighbors = min(k, other_matrix.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(other_matrix)

    src_out, cand_out, score_out = [], [], []
    for start in range(0, s1_matrix.shape[0], query_batch_size):
        end = min(start + query_batch_size, s1_matrix.shape[0])
        batch = s1_matrix[start:end]
        # a row with no n-grams (all-zero vector) has undefined cosine distance;
        # cosine metric handles this as distance=1 (similarity 0), which is
        # correctly filtered out by min_similarity below.
        distances, indices = nn.kneighbors(batch, n_neighbors=n_neighbors)
        similarity = 1.0 - distances
        keep = similarity >= min_similarity
        rows, cols = np.nonzero(keep)
        if rows.size == 0:
            continue
        src_out.append(s1_ids[start:end][rows])
        cand_out.append(other_ids[indices[rows, cols]])
        score_out.append(similarity[rows, cols])

    if not src_out:
        return (np.array([], dtype=object), np.array([], dtype=object), np.array([], dtype=np.float32))
    return np.concatenate(src_out), np.concatenate(cand_out), np.concatenate(score_out)


def tfidf_candidate_pairs(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    text_col: str,
    channel_name: str,
    k: int = 20,
    ngram_range: tuple[int, int] = (3, 4),
    max_features: int = 20000,
    min_similarity: float = 0.2,
    query_batch_size: int = 2000,
    max_df: float = 0.2,
    min_df: int = 2,
) -> pd.DataFrame:
    """Top-k char-ngram TF-IDF cosine candidates per S1 row, per country partition."""
    results = []
    countries = sorted(set(s1_df["country"].dropna().astype(str)) & set(other_df["country"].dropna().astype(str)))
    for country in countries:
        s1_part = s1_df[(s1_df["country"] == country) & (s1_df[text_col] != "")]
        other_part = other_df[(other_df["country"] == country) & (other_df[text_col] != "")]
        if s1_part.empty or other_part.empty:
            continue
        logger.info(
            "tfidf[%s/%s]: s1=%d other=%d", channel_name, country, len(s1_part), len(other_part)
        )
        src, cand, score = _tfidf_knn_for_country(
            s1_part["entity_id"].to_numpy(),
            s1_part[text_col].to_numpy(),
            other_part["entity_id"].to_numpy(),
            other_part[text_col].to_numpy(),
            k=k,
            ngram_range=ngram_range,
            max_features=max_features,
            min_similarity=min_similarity,
            query_batch_size=query_batch_size,
            max_df=max_df,
            min_df=min_df,
        )
        if src.size == 0:
            continue
        results.append(
            pd.DataFrame(
                {
                    "source1_entity_id": src,
                    "candidate_entity_id": cand,
                    "channel": channel_name,
                    "score": score,
                }
            )
        )
    if not results:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "channel", "score"])
    return pd.concat(results, ignore_index=True)
