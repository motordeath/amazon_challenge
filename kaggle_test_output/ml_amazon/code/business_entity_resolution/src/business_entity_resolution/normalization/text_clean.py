"""Script-agnostic base text cleaning shared by name and address normalization.

Conservative by design: Unicode NFKC normalization, casefold, punctuation and
whitespace cleanup, ``&`` -> ``and``. Nothing here transliterates a script or
drops digits/tokens, since the EDA flags both as "potentially dangerous"
(destroys the only distinguishing signal in some records, e.g. ``# 2``,
suite numbers, or a Devanagari business name that address-based blocking
still needs verbatim for later exact-address checks).
"""
from __future__ import annotations

import functools
import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_LEADING_HASH_RE = re.compile(r"^[#\-\s]+")

# Unicode general-category groups to KEEP as-is: Letters (L*), Marks (M*—
# critical for Devanagari matras/virama, which are combining marks and would
# otherwise be silently dropped by a \w-based regex), Numbers (N*).
_KEEP_CATEGORY_PREFIXES = ("L", "M", "N")


@functools.lru_cache(maxsize=8192)
def _scrub_char(ch: str) -> str:
    if ch.isspace():
        return ch
    if unicodedata.category(ch)[0] in _KEEP_CATEGORY_PREFIXES:
        return ch
    return " "


def base_clean(text: str) -> str:
    """NFKC-normalize, casefold, tidy punctuation/whitespace. Returns "" for NaN-like input.

    Punctuation/symbol stripping is done per-character via Unicode general
    category (keeping Letters/Marks/Numbers) rather than a ``\\w`` regex,
    because ``\\w`` does not reliably include combining marks such as
    Devanagari vowel signs — a regex-based strip would silently corrupt
    non-Latin script text.
    """
    if text is None:
        return ""
    text = str(text)
    if not text or text.lower() in {"nan", "n/a", "none"}:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = text.replace("&", " and ")
    text = _LEADING_HASH_RE.sub("", text)
    text = "".join(_scrub_char(ch) for ch in text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return text.split(" ")


def compact_form(tokens: list[str]) -> str:
    """Word-order-invariant representation: sorted tokens joined."""
    return " ".join(sorted(tokens))


def char_ngrams(text: str, n: int = 4) -> list[str]:
    """Character n-grams over the whitespace-padded string (script-agnostic)."""
    if not text:
        return []
    padded = f" {text} "
    if len(padded) < n:
        return [padded]
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]
