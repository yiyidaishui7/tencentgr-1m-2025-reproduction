from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
CONTRACT_PATH = ROOT / "scripts" / "onepiece_evaluation_contract.py"


def load_contract_module():
    spec = importlib.util.spec_from_file_location(
        "onepiece_evaluation_contract", CONTRACT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cold_candidate_filter_preserves_official_retrieval_ids():
    contract = load_contract_module()
    internal = np.array([7, 0, 2, 0, 9], dtype=np.int32)
    retrieval = np.array([100, 101, 102, 103, 104], dtype=np.uint64)
    features = {"100": np.array([1, 2, 3, 4, 5], dtype=np.int32)}

    warm_internal, warm_retrieval, warm_features, cold_count = (
        contract.filter_cold_candidates(internal, retrieval, features)
    )

    np.testing.assert_array_equal(warm_internal, [7, 2, 9])
    np.testing.assert_array_equal(warm_retrieval, [100, 102, 104])
    np.testing.assert_array_equal(warm_features["100"], [1, 3, 5])
    assert cold_count == 2


def test_history_mask_ignores_padding_and_deduplicates_seen_items():
    contract = load_contract_module()
    candidate_rows = contract.build_candidate_row_index(
        np.array([7, 2, 9], dtype=np.int32), item_vocabulary_size=12
    )
    sequences = np.array(
        [[0, 7, 7, 11], [2, 0, 9, 9]], dtype=np.int32
    )

    batch_rows, item_rows = contract.history_mask_indices(
        sequences, candidate_rows
    )

    assert set(zip(batch_rows.tolist(), item_rows.tolist())) == {
        (0, 0),
        (1, 1),
        (1, 2),
    }


def test_beam_fallback_fills_to_ten_without_sentinel_or_duplicates():
    contract = load_contract_module()
    primary = np.array([[4, -1, 4, 8, -1, 6]], dtype=np.int32)
    fallback = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int32)

    merged = contract.merge_ranked_candidate_rows(primary, fallback, output_size=10)

    assert merged.shape == (1, 10)
    assert merged[0].tolist() == [4, 8, 6, 0, 1, 2, 3, 5, 7, 9]
    assert len(set(merged[0].tolist())) == 10
    assert int(merged.min()) >= 0


def test_beam_fallback_is_exact_ann_when_no_legal_sid_exists():
    contract = load_contract_module()
    primary = np.full((2, 10), -1, dtype=np.int32)
    fallback = np.tile(np.arange(10, dtype=np.int32), (2, 1))

    np.testing.assert_array_equal(
        contract.merge_ranked_candidate_rows(primary, fallback), fallback
    )


def test_targets_must_remain_in_the_warm_candidate_pool():
    contract = load_contract_module()

    contract.validate_targets_in_candidate_pool(
        np.array([3, 5], dtype=np.uint64),
        np.array([1, 3, 5], dtype=np.uint64),
    )
    with pytest.raises(RuntimeError, match="target retrieval IDs are absent"):
        contract.validate_targets_in_candidate_pool(
            np.array([3, 8], dtype=np.uint64),
            np.array([1, 3, 5], dtype=np.uint64),
        )


def test_public_runner_applies_the_evaluation_contract_before_topk():
    source = (ROOT / "scripts" / "run_onepiece_formal.py").read_text(
        encoding="utf-8"
    )

    assert "import onepiece_evaluation_contract as evaluation_contract" in source
    assert "filter_cold_candidates" in source
    assert "history_mask_indices" in source
    assert "merge_ranked_candidate_rows" in source
    assert '"history_filtering": True' in source
    assert "np.iinfo(np.uint64).max" not in source
