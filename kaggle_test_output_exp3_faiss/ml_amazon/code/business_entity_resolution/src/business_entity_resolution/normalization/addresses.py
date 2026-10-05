"""Address normalization producing multiple preserved representations.

``address_raw``        - original string, untouched
``address_normalized``  - cleaned, casefolded, street/landmark canonicalized
``address_tokens``      - whitespace tokens of address_normalized
``address_compact``     - tokens sorted then joined (component-order invariant)
``address_numbers``     - numeric tokens extracted (house/flat/PIN/ZIP numbers)
``address_ngrams``      - character 4-grams (typo-robust)
``address_missing``     - True if raw was NaN/empty (~3.3% of S2/S3)
"""
from __future__ import annotations

import re

from .address_abbrev import canonicalize_address_token
from .text_clean import base_clean, char_ngrams, compact_form, tokenize

NGRAM_N = 4
_NUMBER_RE = re.compile(r"\d+")


def normalize_address(raw, compute_ngrams: bool = False) -> dict:
    """See :func:`names.normalize_name` for why ``compute_ngrams`` defaults False."""
    missing = raw is None or (isinstance(raw, float) and raw != raw) or str(raw).strip() == ""
    cleaned = base_clean(raw)
    tokens = tokenize(cleaned)
    canon_tokens = [canonicalize_address_token(t) for t in tokens]
    normalized = " ".join(canon_tokens)
    numbers = _NUMBER_RE.findall(normalized)

    return {
        "address_raw": "" if raw is None else str(raw),
        "address_normalized": normalized,
        "address_tokens": canon_tokens,
        "address_compact": compact_form(canon_tokens),
        "address_numbers": frozenset(numbers),
        "address_ngrams": char_ngrams(normalized, NGRAM_N) if compute_ngrams else None,
        "address_missing": missing,
    }
