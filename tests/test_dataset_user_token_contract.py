from __future__ import annotations

import sys
import types

import numpy as np


torch_was_stubbed = False
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_was_stubbed = True
    torch_stub = types.ModuleType("torch")
    torch_stub.utils = types.SimpleNamespace(
        data=types.SimpleNamespace(Dataset=object),
    )
    sys.modules["torch"] = torch_stub


from dataset import MyDataset, MyTestDataset

if torch_was_stubbed:
    del sys.modules["torch"]


def _dataset_with(records: list[tuple], *, maxlen: int = 4) -> MyDataset:
    dataset = object.__new__(MyDataset)
    dataset.maxlen = maxlen
    dataset.itemnum = 99
    dataset.item_feat_dict = {"99": {"negative": True}}
    dataset.feature_default_value = {"default": 0}
    dataset.new_load_user_data = lambda uid: records
    dataset.fill_missing_feat = lambda feat, token_id: (
        {"filled_for": token_id} if feat is None else dict(feat)
    )
    dataset._random_neq = lambda lower, upper, seen: 99
    return dataset


def _test_dataset_with(records: list[tuple], *, maxlen: int = 4) -> MyTestDataset:
    dataset = object.__new__(MyTestDataset)
    dataset.maxlen = maxlen
    dataset.itemnum = 99
    dataset.indexer_u_rev = {1: "user_1"}
    dataset.feature_default_value = {"default": 0}
    dataset.new_load_user_data = lambda uid: records
    dataset.fill_missing_feat = lambda feat, token_id: (
        {"filled_for": token_id} if feat is None else dict(feat)
    )
    return dataset


def test_training_sequence_has_one_featured_user_token_and_keeps_featureless_item() -> None:
    user_features = {"103": 7, "106": [2, 3]}
    dataset = _dataset_with(
        [
            (1, 11, None, {}, 1, 100),
            (1, 12, None, {"100": 5}, 2, 200),
            (1, None, user_features, None, None, 200),
        ]
    )

    seq, pos, _, token_type, next_token_type, _, seq_feat, _, _ = dataset[0]

    np.testing.assert_array_equal(seq, np.array([0, 0, 0, 1, 11]))
    np.testing.assert_array_equal(token_type, np.array([0, 0, 0, 2, 1]))
    np.testing.assert_array_equal(pos, np.array([0, 0, 0, 11, 12]))
    np.testing.assert_array_equal(next_token_type, np.array([0, 0, 0, 1, 1]))
    assert np.count_nonzero(token_type == 2) == 1
    assert seq_feat[3] == user_features
    assert seq_feat[4] == {}


def test_user_token_contract_at_maxlen_boundary() -> None:
    user_features = {"103": 9}
    records = [
        (1, item_id, None, {"item": item_id}, 1, item_id * 10)
        for item_id in (11, 12, 13, 14, 15)
    ]
    records.append((1, None, user_features, None, None, 150))
    dataset = _dataset_with(records)

    seq, _, _, token_type, _, _, seq_feat, _, _ = dataset[0]

    np.testing.assert_array_equal(seq, np.array([1, 11, 12, 13, 14]))
    np.testing.assert_array_equal(token_type, np.array([2, 1, 1, 1, 1]))
    assert np.count_nonzero(token_type == 2) == 1
    assert seq_feat[0] == user_features


def test_training_keeps_user_token_when_feature_row_is_empty() -> None:
    dataset = _dataset_with(
        [
            (1, 11, None, {}, 1, 100),
            (1, 12, None, {"100": 5}, 2, 200),
            (1, None, {}, None, None, 200),
        ]
    )

    seq, _, _, token_type, _, _, seq_feat, _, _ = dataset[0]

    np.testing.assert_array_equal(seq, np.array([0, 0, 0, 1, 11]))
    np.testing.assert_array_equal(token_type, np.array([0, 0, 0, 2, 1]))
    assert seq_feat[3] == {}


def test_inference_keeps_empty_feature_user_and_item_tokens() -> None:
    dataset = _test_dataset_with(
        [
            (1, 11, None, {}, 1, 100),
            (1, 12, None, {"100": 5}, 2, 200),
            (1, None, {}, None, None, 200),
        ]
    )

    seq, token_type, seq_feat, user_id = dataset[0]

    np.testing.assert_array_equal(seq, np.array([0, 0, 0, 1, 11]))
    np.testing.assert_array_equal(token_type, np.array([0, 0, 0, 2, 1]))
    assert seq_feat[3] == {}
    assert seq_feat[4] == {}
    assert user_id == "user_1"
