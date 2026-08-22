from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HoldoutTarget:
    """A click target and the strictly earlier history boundary."""

    target_position: int
    target_item_id: int
    prefix_length: int


def select_last_click_holdout(
    item_ids: np.ndarray,
    action_types: np.ndarray,
    *,
    click_action: int = 1,
) -> HoldoutTarget | None:
    """Select the last valid clicked item without exposing it to the query history."""

    item_ids = np.asarray(item_ids)
    action_types = np.asarray(action_types)
    if item_ids.shape != action_types.shape:
        raise ValueError("item_ids and action_types must have the same shape")
    if item_ids.ndim != 1:
        raise ValueError("item_ids and action_types must be one-dimensional")

    positions = np.flatnonzero((action_types == click_action) & (item_ids != 0))
    if positions.size == 0:
        return None

    target_position = int(positions[-1])
    return HoldoutTarget(
        target_position=target_position,
        target_item_id=int(item_ids[target_position]),
        prefix_length=target_position,
    )


def competition_metrics(
    topk_ids: np.ndarray,
    target_ids: np.ndarray,
    *,
    k: int = 10,
) -> dict[str, int | float]:
    """Compute TAAC preliminary-round HR@10, NDCG@10, and weighted score."""

    topk_ids = np.asarray(topk_ids)
    target_ids = np.asarray(target_ids)
    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must be a two-dimensional array")
    if target_ids.ndim != 1:
        raise ValueError("target_ids must be a one-dimensional array")
    if topk_ids.shape[0] == 0:
        raise ValueError("metrics require at least one user")
    if target_ids.shape[0] != topk_ids.shape[0]:
        raise ValueError("target count must match prediction rows")
    if not 1 <= k <= topk_ids.shape[1]:
        raise ValueError("k must be between 1 and the prediction width")

    matches = topk_ids[:, :k] == target_ids[:, None]
    hit_mask = matches.any(axis=1)
    first_ranks = matches.argmax(axis=1) + 1
    gains = np.zeros(topk_ids.shape[0], dtype=np.float64)
    gains[hit_mask] = 1.0 / np.log2(first_ranks[hit_mask] + 1.0)

    hits = int(hit_mask.sum())
    hit_rate = float(hit_mask.mean())
    ndcg = float(gains.mean())
    return {
        "evaluated_users": int(topk_ids.shape[0]),
        "hits": hits,
        "hit_rate_at_10": hit_rate,
        "ndcg_at_10": ndcg,
        "competition_score": 0.31 * hit_rate + 0.69 * ndcg,
    }
