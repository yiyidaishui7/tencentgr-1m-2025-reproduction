"""Freeze the row-aligned TencentGR-1M offline evaluation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--retrieval-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.predictions) < 2:
        raise RuntimeError("at least two prediction files are required for row-alignment audit")
    args.output.mkdir(parents=True, exist_ok=True)
    aligned_keys = ("target_retrieval_ids", "prefix_lengths", "user_reids")
    references: dict[str, np.ndarray] = {}
    source_hashes = {}
    for index, path in enumerate(args.predictions):
        source_hashes[str(path)] = sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            current = {key: archive[key] for key in aligned_keys}
        if index == 0:
            references = current
        else:
            for key in aligned_keys:
                if not np.array_equal(references[key], current[key]):
                    raise RuntimeError(f"row alignment mismatch for {key}: {path}")

    retrieval_to_creative = json.loads(args.retrieval_map.read_text(encoding="utf-8"))
    count = len(retrieval_to_creative)
    expected_keys = {str(value) for value in range(count)}
    if set(retrieval_to_creative) != expected_keys:
        raise RuntimeError("retrieval map is not the contiguous range [0, N)")
    candidate_retrieval_ids = np.arange(count, dtype=np.uint64)
    candidate_raw_creative_ids = np.asarray(
        [int(retrieval_to_creative[str(value)]) for value in range(count)],
        dtype=np.uint64,
    )
    if len(np.unique(candidate_raw_creative_ids)) != count:
        raise RuntimeError("candidate creative IDs are not unique")
    target_retrieval_ids = references["target_retrieval_ids"].astype(np.uint64)
    target_raw_creative_ids = candidate_raw_creative_ids[target_retrieval_ids]

    contract_path = args.output / "offline_contract.npz"
    np.savez_compressed(
        contract_path,
        eval_user_reids=references["user_reids"].astype(np.int32),
        eval_target_retrieval_ids=target_retrieval_ids,
        eval_target_raw_creative_ids=target_raw_creative_ids,
        eval_prefix_lengths=references["prefix_lengths"].astype(np.int16),
        candidate_retrieval_ids=candidate_retrieval_ids,
        candidate_raw_creative_ids=candidate_raw_creative_ids,
    )
    metadata = {
        "status": "pass",
        "seed": 2025,
        "dataset_users": 1_001_845,
        "training_users": 901_661,
        "validation_users": 100_184,
        "eligible_eval_users": int(len(target_retrieval_ids)),
        "candidate_count": int(count),
        "candidate_retrieval_range": [0, int(count - 1)],
        "contract_sha256": sha256(contract_path),
        "retrieval_map_sha256": sha256(args.retrieval_map),
        "source_prediction_sha256": source_hashes,
        "row_array_sha256": {
            key: array_sha256(value) for key, value in references.items()
        },
        "target_raw_creative_ids_sha256": array_sha256(target_raw_creative_ids),
        "candidate_raw_creative_ids_sha256": array_sha256(candidate_raw_creative_ids),
        "evaluation": {
            "target": "last action_type=1 item",
            "history": "events strictly before the held-out click",
            "metric": "0.31 * HR@10 + 0.69 * NDCG@10",
        },
    }
    metadata_path = args.output / "protocol_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "SHA256SUMS").write_text(
        f"{sha256(contract_path)}  {contract_path.name}\n"
        f"{sha256(metadata_path)}  {metadata_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
