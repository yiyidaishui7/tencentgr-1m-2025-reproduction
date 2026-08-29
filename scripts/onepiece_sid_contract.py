"""Executable invariants for the SID hierarchy described by OnePiece's README."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _aligned_codes(first: np.ndarray, second: np.ndarray):
    coarse = np.asarray(first, dtype=np.int64)
    fine = np.asarray(second, dtype=np.int64)
    if coarse.ndim != 1 or fine.ndim != 1 or coarse.shape != fine.shape:
        raise ValueError("SID arrays must be aligned one-dimensional arrays")
    if len(coarse) == 0:
        raise ValueError("SID arrays are empty")
    return coarse, fine


def reallocate_fine_codes_preserving_sid1(
    first: np.ndarray,
    second: np.ndarray,
    neighbour_orders_by_l1: Mapping[int, np.ndarray],
    *,
    codebook_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Resolve L2 collisions without ever moving an item across L1.

    Each L1 receives its own L2-neighbour table. Top-ranked unoccupied siblings
    are tried first; a seeded random unoccupied sibling is used only when that
    table is exhausted. Capacity violations fail closed because no legal
    fixed-L1, collision-free mapping can then exist.
    """
    coarse, initial_fine = _aligned_codes(first, second)
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    if int(coarse.min()) < 0 or int(initial_fine.min()) < 0:
        raise RuntimeError("SID values must be non-negative")
    if int(initial_fine.max()) >= codebook_size:
        raise RuntimeError("initial SID2 is outside the fine codebook")
    coarse_counts = np.bincount(coarse)
    overflowing = np.flatnonzero(coarse_counts > codebook_size)
    if len(overflowing):
        group = int(overflowing[0])
        raise RuntimeError(
            f"L1 {group} has {int(coarse_counts[group])} items and exceeds "
            f"fine-code capacity {codebook_size}"
        )

    output = np.empty_like(initial_fine, dtype=np.int32)
    retained = 0
    top_k_reassigned = 0
    fallback_reassigned = 0
    for group in np.unique(coarse):
        group = int(group)
        neighbour_order = np.asarray(neighbour_orders_by_l1.get(group))
        if neighbour_order.ndim != 2:
            raise RuntimeError(f"missing L1-scoped L2 neighbour table for {group}")
        if bool((neighbour_order < 0).any()) or bool(
            (neighbour_order >= codebook_size).any()
        ):
            raise RuntimeError(f"L2 neighbour table is outside the codebook for L1 {group}")
        rows = np.flatnonzero(coarse == group)
        used = np.zeros(codebook_size, dtype=np.bool_)
        collisions = []
        for row in rows:
            code = int(initial_fine[row])
            if not used[code]:
                output[row] = code
                used[code] = True
                retained += 1
            else:
                collisions.append(int(row))

        generator = np.random.default_rng(np.random.SeedSequence([seed, group]))
        for row in collisions:
            original = int(initial_fine[row])
            if original >= neighbour_order.shape[0]:
                raise RuntimeError(
                    f"L2 neighbour table lacks original code {original} for L1 {group}"
                )
            assigned = -1
            for candidate in neighbour_order[original]:
                candidate = int(candidate)
                if not used[candidate]:
                    assigned = candidate
                    top_k_reassigned += 1
                    break
            if assigned < 0:
                available = np.flatnonzero(~used)
                available = available[available != original]
                if len(available) == 0:
                    raise RuntimeError(f"fine-code capacity exhausted for L1 {group}")
                assigned = int(generator.choice(available))
                fallback_reassigned += 1
            output[row] = assigned
            used[assigned] = True

    audit = validate_collision_free_mapping(
        coarse,
        coarse.copy(),
        output,
        coarse_codebook_size=int(coarse.max()) + 1,
        fine_codebook_size=codebook_size,
    )
    audit.update(
        {
            "original_pair_retained": retained,
            "top_k_reassigned": top_k_reassigned,
            "fallback_reassigned": fallback_reassigned,
        }
    )
    return output, audit


def validate_collision_free_mapping(
    initial_first: np.ndarray,
    final_first: np.ndarray,
    final_second: np.ndarray,
    coarse_codebook_size: int,
    fine_codebook_size: int,
) -> dict[str, int]:
    initial_coarse, fine = _aligned_codes(initial_first, final_second)
    final_coarse = np.asarray(final_first, dtype=np.int64)
    if final_coarse.ndim != 1 or final_coarse.shape != initial_coarse.shape:
        raise ValueError("final SID1 array is not aligned")
    sid1_changed = int((initial_coarse != final_coarse).sum())
    if sid1_changed:
        raise RuntimeError(f"SID1 changed for {sid1_changed} items")
    if coarse_codebook_size <= 0 or fine_codebook_size <= 0:
        raise ValueError("codebook sizes must be positive")
    if (
        int(final_coarse.min()) < 0
        or int(final_coarse.max()) >= coarse_codebook_size
        or int(fine.min()) < 0
        or int(fine.max()) >= fine_codebook_size
    ):
        raise RuntimeError("final SID mapping falls outside its codebooks")
    pair_keys = final_coarse * fine_codebook_size + fine
    unique_keys, reverse_rows = np.unique(pair_keys, return_index=True)
    collisions = len(pair_keys) - len(unique_keys)
    if collisions:
        raise RuntimeError(f"final SID mapping contains {collisions} pair collisions")
    reverse = {int(key): int(row) for key, row in zip(unique_keys, reverse_rows)}
    roundtrip_mismatch = sum(
        reverse[int(key)] != row for row, key in enumerate(pair_keys)
    )
    if roundtrip_mismatch:
        raise RuntimeError("SID reverse lookup does not round-trip")
    return {
        "items": len(pair_keys),
        "sid1_changed": sid1_changed,
        "pair_collision_count": collisions,
        "reverse_roundtrip_mismatch": roundtrip_mismatch,
        "max_l1_size": int(np.bincount(final_coarse).max()),
    }
