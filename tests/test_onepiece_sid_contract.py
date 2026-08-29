from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
CONTRACT_PATH = ROOT / "scripts" / "onepiece_sid_contract.py"


def load_contract_module():
    spec = importlib.util.spec_from_file_location(
        "onepiece_sid_contract", CONTRACT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reallocator_preserves_sid1_or_fails_capacity():
    contract = load_contract_module()
    with pytest.raises(RuntimeError, match="exceeds fine-code capacity"):
        contract.reallocate_fine_codes_preserving_sid1(
            np.array([0, 0, 0], dtype=np.int32),
            np.array([0, 0, 1], dtype=np.int32),
            {0: np.array([[0, 1], [1, 0]], dtype=np.int32)},
            codebook_size=2,
            seed=7,
        )


def test_l2_neighbour_search_is_scoped_by_l1():
    contract = load_contract_module()
    first = np.array([0, 0, 1, 1], dtype=np.int32)
    second = np.array([0, 0, 0, 0], dtype=np.int32)
    neighbours = {
        0: np.array([[1, 0], [0, 1]], dtype=np.int32),
        1: np.array([[2, 0], [1, 0], [0, 2]], dtype=np.int32),
    }

    fine, audit = contract.reallocate_fine_codes_preserving_sid1(
        first, second, neighbours, codebook_size=3, seed=11
    )

    np.testing.assert_array_equal(fine, [0, 1, 0, 2])
    assert audit["sid1_changed"] == 0
    contract.validate_collision_free_mapping(first, first.copy(), fine, 2, 3)


def test_random_fallback_is_seeded_unoccupied_and_same_l1():
    contract = load_contract_module()
    first = np.zeros(4, dtype=np.int32)
    second = np.zeros(4, dtype=np.int32)
    neighbours = {0: np.array([[0], [1], [2], [3]], dtype=np.int32)}

    first_run, first_audit = contract.reallocate_fine_codes_preserving_sid1(
        first, second, neighbours, codebook_size=4, seed=20250825
    )
    second_run, second_audit = contract.reallocate_fine_codes_preserving_sid1(
        first, second, neighbours, codebook_size=4, seed=20250825
    )

    np.testing.assert_array_equal(first_run, second_run)
    assert len(np.unique(first_run)) == 4
    assert first_audit == second_audit
    assert first_audit["fallback_reassigned"] == 3


def test_mapping_reverse_lookup_is_bijective_and_sid1_change_fails():
    contract = load_contract_module()
    initial_first = np.array([0, 0, 1], dtype=np.int32)
    final_first = initial_first.copy()
    final_second = np.array([0, 1, 0], dtype=np.int32)

    audit = contract.validate_collision_free_mapping(
        initial_first, final_first, final_second, 2, 2
    )
    assert audit["pair_collision_count"] == 0
    assert audit["reverse_roundtrip_mismatch"] == 0

    final_first[0] = 1
    with pytest.raises(RuntimeError, match="SID1 changed"):
        contract.validate_collision_free_mapping(
            initial_first, final_first, final_second, 2, 2
        )
