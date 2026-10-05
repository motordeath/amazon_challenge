"""Business-name normalization producing multiple preserved representations.

Per the EDA and task spec, one overly-aggressive normalized string would
destroy signal (suffix-only differentiators, script information). We instead
return a dict of representations and let downstream feature code choose
which to compare:

``name_raw``        - original string, untouched
``name_normalized``  - cleaned, casefolded, suffix-canonicalized
``name_tokens``      - whitespace tokens of name_normalized
``name_compact``     - tokens sorted then joined (word-order invariant)
``name_core``        - name_normalized with legal-suffix tokens removed
``name_ngrams``      - character 4-grams of name_normalized (typo-robust)
``name_missing``     - True if the raw value was NaN/empty
"""
from __future__ import annotations

from .legal_suffixes import (
    SUFFIX_CANONICAL_TOKENS as _CANON_SUFFIXES,
    canonicalize_suffix_token,
    is_suffix_token,
)
from .text_clean import base_clean, char_ngrams, compact_form, tokenize

NGRAM_N = 4


def normalize_name(raw, compute_ngrams: bool = False) -> dict:
    """``compute_ngrams`` defaults to False: n-grams are only needed for the
    (much smaller) candidate-pair feature matrix, not for every one of the
    ~12.5M corpus rows touched during blocking — computing/storing an extra
    20-30-element list per row there is a significant, avoidable memory cost.
    """
    missing = raw is None or (isinstance(raw, float) and raw != raw) or str(raw).strip() == ""
    cleaned = base_clean(raw)
    tokens = tokenize(cleaned)
    canon_tokens = [canonicalize_suffix_token(t) for t in tokens]
    normalized = " ".join(canon_tokens)
    core_tokens = [t for t in canon_tokens if not is_suffix_token(t) and t not in _CANON_SUFFIXES]
    suffix_tokens = [t for t in canon_tokens if t in _CANON_SUFFIXES]

    return {
        "name_raw": "" if raw is None else str(raw),
        "name_normalized": normalized,
        "name_tokens": canon_tokens,
        "name_compact": compact_form(canon_tokens),
        "name_core": " ".join(core_tokens),
        "name_suffixes": frozenset(suffix_tokens),
        "name_ngrams": char_ngrams(normalized, NGRAM_N) if compute_ngrams else None,
        "name_missing": missing,
    }
