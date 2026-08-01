"""General record normalization before blocking and matching."""

from __future__ import annotations

import html
import math
import re
import unicodedata

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
    if any(token in col for token in ("modelno", "model_no", "model_number", "model", "sku", "upc")):
        return "model_rich"
    if any(token in col for token in ("price", "cost", "msrp")):
        return "price"
    if any(token in col for token in ("shipweight", "shipping_weight", "weight")):
        return "weight"
    if any(token in col for token in ("dimensions", "dimension", "size")):
        return "dimensions"
    if any(token in col for token in ("phone", "telephone", "tel", "mobile", "fax")):
        return "phone"
    if any(token in col for token in ("addr", "address", "street", "location")):
        return "address"
    return "text"


def _decode_text(value) -> str:
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def normalize_text_value(value) -> str:
    """Normalize generic text while preserving the original information."""
    if _is_missing(value):
        return ""

    text = _decode_text(value).strip().lower()
    text = text.replace("`", "'").replace('"', " ")
    text = re.sub(r"\\+'", "'", text)
    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"[^\w#&/.'+-]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip(" '")
    return text


def normalize_phone_value(value) -> str:
    """Canonicalize phone-like values by keeping their digits."""
    if _is_missing(value):
        return ""

    digits = re.sub(r"\D+", "", str(value))
    return digits


def normalize_model_value(value) -> str:
    """Normalize product identifiers without discarding the raw structure."""
    raw = normalize_text_value(value)
    if not raw:
        return ""

    canonical = re.sub(r"[^a-z0-9]+", "", raw)
    if not canonical:
        return raw
    return f"{raw} | norm:{canonical}"


def normalize_price_value(value) -> str:
    """Normalize simple price-like values to comparable numeric text."""
    if _is_missing(value):
        return ""

    text = _decode_text(value).strip().lower()
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not match:
        return normalize_text_value(value)
    number = match.group(0).replace(",", "")
    try:
        number_value = float(number)
    except ValueError:
        return normalize_text_value(value)
    return f"{number_value:.10g}"


def _format_number(value: float) -> str:
    return f"{value:.10g}"


def normalize_weight_value(value) -> str:
    """Normalize simple pound/ounce weight variants."""
    if _is_missing(value):
        return ""

    text = normalize_text_value(value)
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(pounds?|lbs?|ounces?|oz)\b", text)
    if not match:
        return text

    amount = float(match.group(1))
    unit = match.group(2)
    pounds = amount / 16.0 if unit in {"oz", "ounce", "ounces"} else amount
    return f"{_format_number(pounds)} lb"


def normalize_dimensions_value(value) -> str:
    """Normalize simple product dimension separators and inch markers."""
    if _is_missing(value):
        return ""

    text = normalize_text_value(value)
    unit_matches = re.findall(r"\b(?:inches|inch|in|centimeters|centimeter|cm|millimeters|millimeter|mm|feet|foot|ft)\b", text)
    unit_map = {
        "inches": "in",
        "inch": "in",
        "in": "in",
        "centimeters": "cm",
        "centimeter": "cm",
        "cm": "cm",
        "millimeters": "mm",
        "millimeter": "mm",
        "mm": "mm",
        "feet": "ft",
        "foot": "ft",
        "ft": "ft",
    }
    units = {unit_map[unit] for unit in unit_matches}
    unit = next(iter(units)) if len(units) == 1 else "in"

    text = re.sub(r"\b(?:inches|inch|in)\b", " in ", text)
    text = re.sub(r"\b(?:centimeters|centimeter|cm)\b", " cm ", text)
    text = re.sub(r"\b(?:millimeters|millimeter|mm)\b", " mm ", text)
    text = re.sub(r"\b(?:feet|foot|ft)\b", " ft ", text)
    text = re.sub(r"(?<=\d)[\"']+", " in ", text)
    text = re.sub(r"\bby\b", " x ", text)
    text = re.sub(r"(?<=\d)\s*[*x]\s*(?=\d)", " x ", text)
    text = re.sub(r"\s+", " ", text).strip()

    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if len(numbers) >= 2:
        return " x ".join(_format_number(float(number)) for number in numbers[:3]) + f" {unit}"
    return text


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


def normalize_record_value(value, column_name: str, normalization_roles=None) -> str:
    """Normalize one cell using a conservative role inferred from its column."""
    normalization_roles = normalization_roles or {}
    role = normalization_roles.get(column_name, _column_role(column_name))
    if role == "phone":
        return normalize_phone_value(value)
    if role == "address":
        return normalize_address_value(value)
    if role == "model_rich":
        return normalize_model_value(value)
    if role == "price":
        return normalize_price_value(value)
    if role == "weight":
        return normalize_weight_value(value)
    if role == "dimensions":
        return normalize_dimensions_value(value)
    return normalize_text_value(value)


def normalize_dataframe_records(df: pd.DataFrame, normalization_roles=None) -> pd.DataFrame:
    """Return a normalized copy of a record dataframe."""
    normalization_roles = normalization_roles or {}
    normalized = df.copy()
    for col in normalized.columns:
        normalized[col] = normalized[col].map(
            lambda value, column=col: normalize_record_value(value, column, normalization_roles)
        )
    return normalized
