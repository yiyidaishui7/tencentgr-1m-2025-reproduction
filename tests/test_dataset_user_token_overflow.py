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

try:
    from dataset import MyDataset, MyTestDataset
finally:
    if torch_was_stubbed:
        sys.modules.pop("dataset", None)
        sys.modules.pop("torch", None)


def _records(item_ids: tuple[int, ...]) -> list[tuple]:
    records = [
        (1, item_id, None, {"item": item_id}, 1, item_id * 10)
        for item_id in item_ids
    ]
    records.append((1, None, {}, None, None, item_ids[-1] * 10))
    return records


def _training_dataset(records: list[tuple]) -> MyDataset:
    dataset = object.__new__(MyDataset)
    dataset.maxlen = 4
    dataset.itemnum = 99
    dataset.item_feat_dict = {"99": {"negative": True}}
    dataset.feature_default_value = {"default": 0}
    dataset.new_load_user_data = lambda uid: records
    dataset.fill_missing_feat = lambda feat, token_id: (
        {"filled_for": token_id} if feat is None else dict(feat)
    )
    dataset._random_neq = lambda lower, upper, seen: 99
    return dataset


def _inference_dataset(records: list[tuple]) -> MyTestDataset:
    dataset = object.__new__(MyTestDataset)
    dataset.maxlen = 4
    dataset.itemnum = 99
    dataset.indexer_u_rev = {1: "user_1"}
    dataset.feature_default_value = {"default": 0}
    dataset.new_load_user_data = lambda uid: records
    dataset.fill_missing_feat = lambda feat, token_id: (
        {"filled_for": token_id} if feat is None else dict(feat)
    )
    return dataset


def test_training_overflow_keeps_user_and_most_recent_history() -> None:
    dataset = _training_dataset(_records((11, 12, 13, 14, 15, 16)))

    seq, _, _, token_type, _, _, _, _, _ = dataset[0]

    np.testing.assert_array_equal(seq, np.array([1, 12, 13, 14, 15]))
    np.testing.assert_array_equal(token_type, np.array([2, 1, 1, 1, 1]))


def test_inference_overflow_keeps_user_and_most_recent_history() -> None:
    dataset = _inference_dataset(_records((11, 12, 13, 14, 15, 16)))

    seq, token_type, _, _ = dataset[0]

    np.testing.assert_array_equal(seq, np.array([1, 12, 13, 14, 15]))
    np.testing.assert_array_equal(token_type, np.array([2, 1, 1, 1, 1]))

