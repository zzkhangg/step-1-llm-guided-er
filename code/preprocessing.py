"""General record normalization before blocking and matching."""

from __future__ import annotations

import math
import re

import pandas as pd


_ADDRESS_ABBREVIATIONS = {
    "avenue": "ave",
    "boulevard": "blvd",
    "circle": "cir",
    "court": "ct",
    "drive": "dr",
    "highway": "hwy",
    "lane": "ln",
    "parkway": "pkwy",
    "place": "pl",
    "road": "rd",
    "square": "sq",
    "street": "st",
    "terrace": "ter",
}

_DIRECTION_ABBREVIATIONS = {
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
}


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return isinstance(value, float) and math.isnan(value)
    except TypeError:
        return False


def _column_role(column_name: str) -> str:
    col = str(column_name).lower()
    if any(token in col for token in ("phone", "telephone", "tel", "mobile", "fax")):
        return "phone"
    if any(token in col for token in ("addr", "address", "street", "location")):
        return "address"
    return "text"


def normalize_text_value(value) -> str:
    """Normalize generic text while preserving the original information."""
    if _is_missing(value):
        return ""

    text = str(value).strip().lower()
    text = text.replace("`", "'").replace('"', " ")
    text = re.sub(r"\\+'", "'", text)
    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"[^a-z0-9#&/.'+-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" '")
    return text


def normalize_phone_value(value) -> str:
    """Canonicalize phone-like values by keeping their digits."""
    if _is_missing(value):
        return ""

    digits = re.sub(r"\D+", "", str(value))
    return digits


def normalize_address_value(value) -> str:
    """Normalize common address punctuation and abbreviations."""
    text = normalize_text_value(value)
    if not text:
        return ""

    text = re.sub(r"[.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = []
    for token in text.split():
        bare = token.strip(".")
        bare = _ADDRESS_ABBREVIATIONS.get(bare, bare)
        bare = _DIRECTION_ABBREVIATIONS.get(bare, bare)
        tokens.append(bare)
    return " ".join(tokens)


def normalize_record_value(value, column_name: str) -> str:
    """Normalize one cell using a conservative role inferred from its column."""
    role = _column_role(column_name)
    if role == "phone":
        return normalize_phone_value(value)
    if role == "address":
        return normalize_address_value(value)
    return normalize_text_value(value)


def normalize_dataframe_records(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of a record dataframe."""
    normalized = df.copy()
    for col in normalized.columns:
        normalized[col] = normalized[col].map(lambda value, column=col: normalize_record_value(value, column))
    return normalized
