"""Pure post-processing contracts for OnePiece-aligned candidate evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def filter_cold_candidates(
    internal_ids: np.ndarray,
    retrieval_ids: np.ndarray,
    candidate_features: Mapping[str, np.ndarray],
):
    """Remove candidates absent from the frozen training item index."""
    internal = np.asarray(internal_ids)
    retrieval = np.asarray(retrieval_ids)
    if internal.ndim != 1 or retrieval.ndim != 1 or len(internal) != len(retrieval):
        raise ValueError("candidate ID arrays must be aligned one-dimensional arrays")
    warm = internal > 0
    filtered_features = {}
    for name, values in candidate_features.items():
        array = np.asarray(values)
        if array.ndim != 1 or len(array) != len(internal):
            raise ValueError(f"candidate feature is not row-aligned: {name}")
        filtered_features[name] = array[warm]
    return (
        internal[warm],
        retrieval[warm],
        filtered_features,
        int((~warm).sum()),
    )


def build_candidate_row_index(
    internal_ids: np.ndarray, item_vocabulary_size: int
) -> np.ndarray:
    """Build a dense internal-item-ID to active-candidate-row lookup."""
    internal = np.asarray(internal_ids, dtype=np.int64)
    if internal.ndim != 1 or item_vocabulary_size <= 0:
        raise ValueError("invalid candidate IDs or item vocabulary size")
    if len(internal) == 0:
        raise ValueError("the warm candidate pool is empty")
    if int(internal.min()) < 1 or int(internal.max()) > item_vocabulary_size:
        raise RuntimeError("warm candidate internal IDs fall outside the item vocabulary")
    if len(np.unique(internal)) != len(internal):
        raise RuntimeError("warm candidate internal IDs are not unique")
    row_by_internal = np.full(item_vocabulary_size + 1, -1, dtype=np.int32)
    row_by_internal[internal] = np.arange(len(internal), dtype=np.int32)
    return row_by_internal


def history_mask_indices(
    sequence_internal_ids: np.ndarray, row_by_internal_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique ``(batch row, candidate row)`` pairs to mask as seen."""
    sequences = np.asarray(sequence_internal_ids, dtype=np.int64)
    lookup = np.asarray(row_by_internal_id)
    if sequences.ndim != 2 or lookup.ndim != 1:
        raise ValueError("history sequences must be 2-D and the lookup must be 1-D")
    in_range = (sequences > 0) & (sequences < len(lookup))
    mapped = np.full(sequences.shape, -1, dtype=np.int64)
    mapped[in_range] = lookup[sequences[in_range]]
    batch_rows, positions = np.nonzero(mapped >= 0)
    if len(batch_rows) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    candidate_rows = mapped[batch_rows, positions]
    pair_width = len(lookup)
    unique_pairs = np.unique(batch_rows * pair_width + candidate_rows)
    return (
        (unique_pairs // pair_width).astype(np.int64, copy=False),
        (unique_pairs % pair_width).astype(np.int64, copy=False),
    )


def merge_ranked_candidate_rows(
    primary_rows: np.ndarray,
    fallback_rows: np.ndarray,
    output_size: int = 10,
) -> np.ndarray:
    """Fill a restricted ranking from ANN rows without sentinels or duplicates."""
    primary = np.asarray(primary_rows, dtype=np.int64)
    fallback = np.asarray(fallback_rows, dtype=np.int64)
    if (
        primary.ndim != 2
        or fallback.ndim != 2
        or primary.shape[0] != fallback.shape[0]
        or output_size <= 0
    ):
        raise ValueError("ranked candidate matrices are not batch-aligned")
    merged = np.full((primary.shape[0], output_size), -1, dtype=np.int32)
    for batch_index in range(primary.shape[0]):
        seen = set()
        output = []
        for source in (primary[batch_index], fallback[batch_index]):
            for value in source:
                candidate = int(value)
                if candidate < 0 or candidate in seen:
                    continue
                seen.add(candidate)
                output.append(candidate)
                if len(output) == output_size:
                    break
            if len(output) == output_size:
                break
        if len(output) != output_size:
            raise RuntimeError(
                f"unable to produce {output_size} unique legal candidates for row "
                f"{batch_index}"
            )
        merged[batch_index] = output
    return merged


def count_history_overlaps(
    ranked_candidate_rows: np.ndarray,
    candidate_internal_ids: np.ndarray,
    sequence_internal_ids: np.ndarray,
) -> int:
    ranked = np.asarray(ranked_candidate_rows, dtype=np.int64)
    candidates = np.asarray(candidate_internal_ids, dtype=np.int64)
    sequences = np.asarray(sequence_internal_ids, dtype=np.int64)
    if (
        ranked.ndim != 2
        or sequences.ndim != 2
        or ranked.shape[0] != sequences.shape[0]
        or bool((ranked < 0).any())
        or bool((ranked >= len(candidates)).any())
    ):
        raise ValueError("ranked rows and histories are not aligned legal arrays")
    overlaps = 0
    for batch_index in range(ranked.shape[0]):
        history = sequences[batch_index]
        history = history[history > 0]
        if len(history):
            overlaps += int(
                np.isin(candidates[ranked[batch_index]], history).sum()
            )
    return overlaps


def validate_targets_in_candidate_pool(
    target_retrieval_ids: np.ndarray, candidate_retrieval_ids: np.ndarray
) -> None:
    targets = np.asarray(target_retrieval_ids)
    candidates = np.asarray(candidate_retrieval_ids)
    if targets.ndim != 1 or candidates.ndim != 1:
        raise ValueError("target and candidate retrieval IDs must be one-dimensional")
    missing = ~np.isin(targets, candidates, assume_unique=False)
    if bool(missing.any()):
        examples = targets[missing][:10].tolist()
        raise RuntimeError(
            f"target retrieval IDs are absent from the warm candidate pool: {examples}"
        )
