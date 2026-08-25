"""Verify a completed OnePiece run against its manifest and frozen contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from scripts.compare_onepiece_architectures import load_predictions, row_metrics, sha256, summarize
except ModuleNotFoundError:  # Direct execution via ``python scripts/verify_...py``.
    from compare_onepiece_architectures import load_predictions, row_metrics, sha256, summarize


REQUIRED_FILES = {
    "model.pt",
    "offline_metrics.json",
    "offline_predictions.npz",
    "history.json",
    "config.json",
}


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify_artifact(directory: Path, contract_path: Path) -> dict[str, object]:
    manifest_path = directory / "SHA256SUMS"
    expected: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.strip()] = digest
    missing_manifest_entries = sorted(REQUIRED_FILES - set(expected))
    if missing_manifest_entries:
        raise RuntimeError(f"missing manifest entries: {missing_manifest_entries}")
    missing_files = sorted(name for name in expected if not (directory / name).is_file())
    if missing_files:
        raise RuntimeError(f"missing artifact files: {missing_files}")
    actual = {name: sha256(directory / name) for name in expected}
    mismatches = sorted(name for name in expected if expected[name] != actual[name])
    if mismatches:
        raise RuntimeError(f"SHA-256 mismatch: {mismatches}")

    predictions = load_predictions(directory)
    with np.load(contract_path, allow_pickle=False) as archive:
        contract = {
            "user_reids": archive["eval_user_reids"],
            "target_retrieval_ids": archive["eval_target_retrieval_ids"],
            "prefix_lengths": archive["eval_prefix_lengths"],
        }
    for key, expected_array in contract.items():
        if not np.array_equal(predictions[key], expected_array):
            raise RuntimeError(f"frozen-contract mismatch: {key}")

    recomputed = summarize(row_metrics(predictions))
    recorded = json.loads((directory / "offline_metrics.json").read_text(encoding="utf-8"))["metrics"]
    for key in ("evaluated_users", "hits"):
        if int(recorded[key]) != int(recomputed[key]):
            raise RuntimeError(f"recorded metric mismatch: {key}")
    for key in ("hit_rate_at_10", "ndcg_at_10", "competition_score"):
        if not np.isclose(recorded[key], recomputed[key], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"recorded metric mismatch: {key}")

    return {
        "status": "pass",
        "directory": str(directory),
        "files_verified": len(expected),
        "manifest_sha256": sha256(manifest_path),
        "contract_sha256": sha256(contract_path),
        "row_array_sha256": {
            key: array_sha256(value) for key, value in contract.items()
        },
        "metrics": recomputed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_artifact(args.artifact_dir, args.contract), sort_keys=True))


if __name__ == "__main__":
    main()
