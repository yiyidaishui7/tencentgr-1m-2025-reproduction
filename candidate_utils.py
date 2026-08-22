"""Compatibility helpers for public and legacy candidate schemas."""

from __future__ import annotations

from typing import Any, Collection


def canonical_id_key(value: Any) -> str:
    """Normalize anonymous IDs used as dictionary keys across Parquet readers."""
    return str(value)


def candidate_item_column(columns: Collection[str]) -> str:
    """Return the raw candidate item identifier column."""
    if "creative_id" in columns:
        return "creative_id"
    if "item_id" in columns:
        return "item_id"
    raise ValueError("Candidate data has neither creative_id nor item_id")


def decode_candidate_feature(value: Any) -> int:
    """Decode legacy ``is_str`` and public ``cold_start`` feature structs."""
    if value is None:
        return 0
    if not isinstance(value, dict):
        return int(value)

    is_cold_start = bool(value.get("cold_start", value.get("is_str", 1)))
    if is_cold_start:
        return 0
    raw_value = value.get("feature_value")
    if raw_value is None:
        return 0
    return int(raw_value)
