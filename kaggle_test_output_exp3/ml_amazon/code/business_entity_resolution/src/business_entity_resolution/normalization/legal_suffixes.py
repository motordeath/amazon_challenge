"""Canonical legal-suffix vocabulary.

Per the EDA (``eda_findings.md`` section 11), suffix variants are safe to
canonicalize (``Ltd``/``Limited`` -> ``ltd``) but dangerous to strip outright:
for generic names (``Summit Inc`` vs ``Summit LLC``) the suffix may be the
only distinguishing signal. We therefore expose the mapping AND a helper
that returns the name both with and without suffix tokens, so features can
be computed on both variants instead of destroying the suffix information.
"""
from __future__ import annotations

# variant -> canonical token. Longer/more specific variants first is not
# required here since matching is done on whole whitespace-delimited tokens.
SUFFIX_MAP: dict[str, str] = {
    "ltd": "ltd",
    "limited": "ltd",
    "pvt": "pvt",
    "private": "pvt",
    "corp": "corp",
    "corporation": "corp",
    "inc": "inc",
    "incorporated": "inc",
    "llc": "llc",
    "l.l.c": "llc",
    "llp": "llp",
    "l.l.p": "llp",
    "plc": "plc",
    "gmbh": "gmbh",
    "co": "co",
    "company": "co",
    "sa": "sa",
    "s.a": "sa",
    "bv": "bv",
    "nv": "nv",
    "ag": "ag",
    "kg": "kg",
    "srl": "srl",
    "sarl": "sarl",
}

SUFFIX_CANONICAL_TOKENS = frozenset(SUFFIX_MAP.values())
SUFFIX_VARIANT_TOKENS = frozenset(SUFFIX_MAP.keys())


def canonicalize_suffix_token(token: str) -> str:
    return SUFFIX_MAP.get(token, token)


def is_suffix_token(token: str) -> bool:
    return token in SUFFIX_VARIANT_TOKENS
