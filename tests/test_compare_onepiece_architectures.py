from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scripts.compare_onepiece_architectures import (
    load_predictions,
    parse_named_directory,
    paired_effect,
    row_metrics,
    summarize,
    verify_manifest,
)


def test_metrics_and_manifest_round_trip(tmp_path):
    predictions = tmp_path / "offline_predictions.npz"
    np.savez_compressed(
        predictions,
        user_reids=np.array([11, 12], dtype=np.int32),
        target_retrieval_ids=np.array([7, 8], dtype=np.uint64),
        prefix_lengths=np.array([5, 90], dtype=np.int16),
        topk_retrieval_ids=np.array(
            [[7, 1, 2, 3, 4, 5, 6, 8, 9, 10], [1, 2, 8, 4, 5, 6, 7, 9, 10, 11]],
            dtype=np.uint64,
        ),
    )
    digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(
        f"{digest}  offline_predictions.npz\n", encoding="utf-8"
    )

    assert verify_manifest(tmp_path)["offline_predictions.npz"] == digest
    rows = row_metrics(load_predictions(tmp_path))
    metrics = summarize(rows)
    assert metrics["hit_rate_at_10"] == 1.0
    assert metrics["ndcg_at_10"] == pytest.approx((1.0 + 0.5) / 2)
    assert metrics["competition_score"] == pytest.approx(0.31 + 0.69 * 0.75)


def test_paired_effect_reports_mean_and_interval():
    result = paired_effect(np.array([1.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]))
    assert result["mean_delta"] == pytest.approx(1 / 3)
    assert result["normal_95_low"] < result["mean_delta"] < result["normal_95_high"]


def test_manifest_rejects_modified_file(tmp_path):
    payload = tmp_path / "offline_predictions.npz"
    payload.write_bytes(b"before")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(
        f"{digest}  offline_predictions.npz\n", encoding="utf-8"
    )
    payload.write_bytes(b"after")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_manifest(tmp_path)


def test_remote_checksum_manifest_and_named_directory(tmp_path):
    payload = tmp_path / "offline_predictions.npz"
    payload.write_bytes(b"predictions")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (tmp_path / "remote_checksums.sha256").write_text(
        f"{digest}  offline_predictions.npz\n", encoding="utf-8"
    )

    assert verify_manifest(tmp_path)["offline_predictions.npz"] == digest
    name, directory = parse_named_directory(f"mm101={tmp_path}")
    assert name == "mm101"
    assert directory == tmp_path
