"""Address abbreviation vocabulary observed in the EDA (section 5/11).

Street-type and Indian landmark-preposition variants are canonicalized;
building/flat/PIN numbers are never stripped (addresses depend on them).
"""
from __future__ import annotations

STREET_TYPE_MAP: dict[str, str] = {
    "road": "rd", "rd": "rd",
    "street": "st", "st": "st",
    "avenue": "ave", "ave": "ave",
    "drive": "dr", "dr": "dr",
    "boulevard": "blvd", "blvd": "blvd",
    "lane": "ln", "ln": "ln",
    "circle": "cir", "cir": "cir",
    "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl",
    "terrace": "ter", "ter": "ter",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy",
    "square": "sq", "sq": "sq",
    "trail": "trl", "trl": "trl",
}

LANDMARK_MAP: dict[str, str] = {
    "opposite": "opp", "opp": "opp",
    "near": "nr", "nr": "nr",
    "floor": "flr", "flr": "flr",
    "behind": "behind",
    "phase": "phase",
    "sector": "sector",
}

UNIT_MAP: dict[str, str] = {
    "apartment": "apt", "apt": "apt",
    "unit": "unit",
    "suite": "ste", "ste": "ste",
    "flat": "flat",
}

ADDRESS_TOKEN_MAP: dict[str, str] = {**STREET_TYPE_MAP, **LANDMARK_MAP, **UNIT_MAP}


def canonicalize_address_token(token: str) -> str:
    return ADDRESS_TOKEN_MAP.get(token, token)
