"""Batch normalization: apply name/address normalization to a whole source frame.

Memory notes (this got tuned hard after two full-scale OOM kills on this
12.5M-row training set + ~25GB-RAM machine):

* Character n-grams are NOT computed here — only for the much smaller
  candidate-pair subset, via :func:`add_ngram_columns`, right before feature
  engineering.
* Token lists are NOT stored either. ``name_tokens``/``address_tokens`` would
  just be ``name_normalized.split(" ")``/``address_normalized.split(" ")`` —
  storing that as a second, separate Python-object list column per row across
  the full corpus doubles the object overhead for information already
  present in the normalized string. ``blocking/token.py`` splits on demand.
* Plain string outputs are cast to the pyarrow-backed ``string`` dtype
  instead of left as generic Python-object columns: a plain-object string
  column carries substantial per-cell Python overhead (~50 bytes) on top of
  the character data, multiplied over 5-12M rows and several columns.
"""
from __future__ import annotations

import logging

import pandas as pd

from .addresses import normalize_address
from .names import normalize_name
from .text_clean import char_ngrams

logger = logging.getLogger(__name__)

_STRING_DTYPE = "string[pyarrow]"


def normalize_source(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame with normalized name/address columns added.

    No n-grams, no separately-stored token lists (see module docstring) —
    just the strings needed for exact/compact/core comparisons plus the
    small suffix/number sets.
    """
    out = df.copy()

    name_records = out["business_name"].map(normalize_name)
    out["name_normalized"] = name_records.map(lambda d: d["name_normalized"]).astype(_STRING_DTYPE)
    out["name_compact"] = name_records.map(lambda d: d["name_compact"]).astype(_STRING_DTYPE)
    out["name_core"] = name_records.map(lambda d: d["name_core"]).astype(_STRING_DTYPE)
    out["name_suffixes"] = name_records.map(lambda d: d["name_suffixes"])
    out["name_missing"] = name_records.map(lambda d: d["name_missing"])
    del name_records

    addr_records = out["business_address"].map(normalize_address)
    out["address_normalized"] = addr_records.map(lambda d: d["address_normalized"]).astype(_STRING_DTYPE)
    out["address_compact"] = addr_records.map(lambda d: d["address_compact"]).astype(_STRING_DTYPE)
    out["address_numbers"] = addr_records.map(lambda d: d["address_numbers"])
    out["address_missing"] = addr_records.map(lambda d: d["address_missing"])
    del addr_records

    return out


def add_ngram_columns(df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """Add ``name_ngrams``/``address_ngrams`` — call only on the small
    candidate-relevant subset right before feature engineering, never on a
    full source corpus."""
    out = df.copy()
    out["name_ngrams"] = out["name_normalized"].map(lambda s: char_ngrams(s, n))
    out["address_ngrams"] = out["address_normalized"].map(lambda s: char_ngrams(s, n))
    return out
