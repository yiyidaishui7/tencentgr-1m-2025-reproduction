from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.verify_onepiece_artifact import REQUIRED_FILES, verify_artifact


def write_fixture(directory, contract, *, corrupt_metric=False):
    topk = np.array(
        [[7, 1, 2, 3, 4, 5, 6, 8, 9, 10], [1, 2, 8, 4, 5, 6, 7, 9, 10, 11]],
        dtype=np.uint64,
    )
    np.savez_compressed(
        directory / "offline_predictions.npz",
        user_reids=np.array([11, 12], dtype=np.int32),
        target_retrieval_ids=np.array([7, 8], dtype=np.uint64),
        prefix_lengths=np.array([5, 90], dtype=np.int16),
        topk_retrieval_ids=topk,
    )
    metrics = {
        "evaluated_users": 2,
        "hits": 2,
        "hit_rate_at_10": 1.0,
        "ndcg_at_10": 0.75,
        "competition_score": 0.8275 if not corrupt_metric else 0.0,
    }
    (directory / "offline_metrics.json").write_text(
        json.dumps({"metrics": metrics}), encoding="utf-8"
    )
    (directory / "model.pt").write_bytes(b"model")
    (directory / "history.json").write_text("[]", encoding="utf-8")
    (directory / "config.json").write_text("{}", encoding="utf-8")
    manifest = []
    for name in sorted(REQUIRED_FILES):
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        manifest.append(f"{digest}  {name}")
    (directory / "SHA256SUMS").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    np.savez_compressed(
        contract,
        eval_user_reids=np.array([11, 12], dtype=np.int32),
        eval_target_retrieval_ids=np.array([7, 8], dtype=np.uint64),
        eval_prefix_lengths=np.array([5, 90], dtype=np.int16),
    )


def test_verify_artifact_passes_complete_fixture(tmp_path):
    contract = tmp_path / "contract.npz"
    write_fixture(tmp_path, contract)

    result = verify_artifact(tmp_path, contract)

    assert result["status"] == "pass"
    assert result["files_verified"] == 5
    assert result["metrics"]["competition_score"] == pytest.approx(0.8275)


def test_verify_artifact_rejects_recorded_metric_drift(tmp_path):
    contract = tmp_path / "contract.npz"
    write_fixture(tmp_path, contract, corrupt_metric=True)

    with pytest.raises(RuntimeError, match="recorded metric mismatch"):
        verify_artifact(tmp_path, contract)

