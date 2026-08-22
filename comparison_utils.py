"""Row-aligned comparison utilities for the 2x2 recency/MM experiment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


REQUIRED_VARIANTS = ("mm101", "nomm101", "mm50", "nomm50")
ALIGNMENT_FIELDS = ("target_retrieval_ids", "prefix_lengths", "user_reids")


def per_user_metric_components(
    topk_ids: np.ndarray,
    target_ids: np.ndarray,
    *,
    k: int = 10,
) -> dict[str, np.ndarray]:
    """Return per-user HR, NDCG, and weighted-score contributions."""

    topk_ids = np.asarray(topk_ids)
    target_ids = np.asarray(target_ids)
    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must be two-dimensional")
    if target_ids.ndim != 1 or target_ids.shape[0] != topk_ids.shape[0]:
        raise ValueError("target_ids must align with prediction rows")
    if topk_ids.shape[0] == 0:
        raise ValueError("at least one prediction row is required")
    if not 1 <= k <= topk_ids.shape[1]:
        raise ValueError("k must be within the prediction width")

    matches = topk_ids[:, :k] == target_ids[:, None]
    hit_mask = matches.any(axis=1)
    first_ranks = matches.argmax(axis=1) + 1
    hit = hit_mask.astype(np.float64)
    ndcg = np.zeros(topk_ids.shape[0], dtype=np.float64)
    ndcg[hit_mask] = 1.0 / np.log2(first_ranks[hit_mask] + 1.0)
    return {"hit": hit, "ndcg": ndcg, "score": 0.31 * hit + 0.69 * ndcg}


def _validate_and_align(
    predictions: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, int | bool]:
    if set(predictions) != set(REQUIRED_VARIANTS):
        raise ValueError(f"predictions must contain exactly {REQUIRED_VARIANTS}")

    reference = predictions["mm101"]
    required_fields = {"topk_retrieval_ids", *ALIGNMENT_FIELDS}
    for variant, arrays in predictions.items():
        missing = required_fields.difference(arrays)
        if missing:
            raise ValueError(f"{variant} is missing arrays: {sorted(missing)}")
        if np.asarray(arrays["topk_retrieval_ids"]).shape[0] != np.asarray(
            arrays["target_retrieval_ids"]
        ).shape[0]:
            raise ValueError(f"{variant} prediction rows do not align with targets")
        for field in ALIGNMENT_FIELDS:
            if not np.array_equal(np.asarray(arrays[field]), np.asarray(reference[field])):
                raise ValueError(f"row alignment failed for {field} in {variant}")

    evaluated_users = int(np.asarray(reference["target_retrieval_ids"]).shape[0])
    return {
        "evaluated_users": evaluated_users,
        "target_retrieval_ids_equal": True,
        "prefix_lengths_equal": True,
        "user_reids_equal": True,
        "all_equal": True,
    }


def _aggregate(components: Mapping[str, np.ndarray]) -> dict[str, int | float]:
    hit = components["hit"]
    return {
        "evaluated_users": int(hit.shape[0]),
        "hits": int(hit.sum()),
        "hit_rate_at_10": float(hit.mean()),
        "ndcg_at_10": float(components["ndcg"].mean()),
        "competition_score": float(components["score"].mean()),
    }


def _delta_summary(
    reference: Mapping[str, np.ndarray],
    variant: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    score_delta = variant["score"] - reference["score"]
    mean_delta = float(score_delta.mean())
    if score_delta.size > 1:
        margin = 1.96 * float(score_delta.std(ddof=1)) / np.sqrt(score_delta.size)
    else:
        margin = 0.0
    reference_mean = float(reference["score"].mean())
    reference_hit = reference["hit"].astype(bool)
    variant_hit = variant["hit"].astype(bool)
    return {
        "mean_score_delta": mean_delta,
        "relative_score_delta_percent": (
            100.0 * mean_delta / reference_mean if reference_mean != 0.0 else None
        ),
        "score_delta_normal_95ci": [mean_delta - margin, mean_delta + margin],
        "hit_transitions": {
            "both_hit": int((reference_hit & variant_hit).sum()),
            "reference_only_hit": int((reference_hit & ~variant_hit).sum()),
            "variant_only_hit": int((~reference_hit & variant_hit).sum()),
            "neither_hit": int((~reference_hit & ~variant_hit).sum()),
        },
    }


def compare_four_variants(
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    *,
    k: int = 10,
) -> dict[str, Any]:
    """Compare maxlen 101/50 x MM/no-MM with paired uncertainty and interaction."""

    alignment = _validate_and_align(predictions)
    components = {
        name: per_user_metric_components(
            arrays["topk_retrieval_ids"], arrays["target_retrieval_ids"], k=k
        )
        for name, arrays in predictions.items()
    }
    paired = {
        "no_mm_effect_at_101": _delta_summary(components["mm101"], components["nomm101"]),
        "no_mm_effect_at_50": _delta_summary(components["mm50"], components["nomm50"]),
        "window_effect_with_mm": _delta_summary(components["mm101"], components["mm50"]),
        "window_effect_without_mm": _delta_summary(components["nomm101"], components["nomm50"]),
    }
    interaction_delta = (
        components["nomm50"]["score"]
        - components["mm50"]["score"]
        - components["nomm101"]["score"]
        + components["mm101"]["score"]
    )
    interaction = _delta_summary(
        {"score": np.zeros_like(interaction_delta), "hit": np.zeros_like(interaction_delta)},
        {"score": interaction_delta, "hit": np.zeros_like(interaction_delta)},
    )
    interaction.pop("relative_score_delta_percent")
    interaction.pop("hit_transitions")
    return {
        "alignment": alignment,
        "aggregate": {name: _aggregate(value) for name, value in components.items()},
        "paired": paired,
        "interaction": interaction,
        "limitations": [
            "Normal paired intervals describe this fixed-seed evaluation population.",
            "Each configuration was trained once, so training-seed uncertainty is not estimated.",
            "These are offline holdout results, not official leaderboard scores.",
        ],
    }
