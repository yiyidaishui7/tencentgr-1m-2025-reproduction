"""Audit and compare row-aligned HSTU and Transformer recommendation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ALIGNMENT_KEYS = ("user_reids", "target_retrieval_ids", "prefix_lengths")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(directory: Path) -> dict[str, str]:
    manifests = [directory / "SHA256SUMS", directory / "remote_checksums.sha256"]
    manifest = next((path for path in manifests if path.exists()), None)
    if manifest is None:
        raise RuntimeError(f"missing SHA-256 manifest in {directory}")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.strip()] = digest
    required = "offline_predictions.npz"
    if required not in expected or not (directory / required).is_file():
        raise RuntimeError(f"prediction artifact is not covered by manifest in {directory}")
    # Some archived ablations intentionally reference a shared checkpoint and
    # therefore retain its remote checksum without duplicating model.pt.
    actual = {
        name: sha256(directory / name)
        for name in expected
        if (directory / name).is_file()
    }
    mismatches = [name for name in actual if expected[name] != actual[name]]
    if mismatches:
        raise RuntimeError(f"SHA-256 mismatch in {directory}: {mismatches}")
    return actual


def load_predictions(directory: Path) -> dict[str, np.ndarray]:
    with np.load(directory / "offline_predictions.npz", allow_pickle=False) as archive:
        required = (*ALIGNMENT_KEYS, "topk_retrieval_ids")
        missing = [key for key in required if key not in archive]
        if missing:
            raise RuntimeError(f"missing prediction arrays in {directory}: {missing}")
        return {key: archive[key] for key in required}


def row_metrics(predictions: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    topk = predictions["topk_retrieval_ids"]
    targets = predictions["target_retrieval_ids"]
    if topk.ndim != 2 or topk.shape != (len(targets), 10):
        raise RuntimeError(f"unexpected Top-10 shape: {topk.shape}")
    matches = topk == targets[:, None]
    hits = matches.any(axis=1).astype(np.float64)
    ranks = np.where(hits.astype(bool), matches.argmax(axis=1) + 1, 0)
    ndcg = np.zeros(len(targets), dtype=np.float64)
    positive = hits.astype(bool)
    ndcg[positive] = 1.0 / np.log2(ranks[positive] + 1)
    score = 0.31 * hits + 0.69 * ndcg
    return hits, ndcg, score


def summarize(rows: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, float | int]:
    hits, ndcg, score = rows
    return {
        "evaluated_users": int(len(hits)),
        "hits": int(hits.sum()),
        "hit_rate_at_10": float(hits.mean()),
        "ndcg_at_10": float(ndcg.mean()),
        "competition_score": float(score.mean()),
    }


def paired_effect(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = candidate - reference
    mean = float(delta.mean())
    standard_error = float(delta.std(ddof=1) / math.sqrt(len(delta)))
    return {
        "mean_delta": mean,
        "normal_95_low": mean - 1.96 * standard_error,
        "normal_95_high": mean + 1.96 * standard_error,
    }


def parse_named_directory(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or name in {"hstu", "transformer"}:
        raise argparse.ArgumentTypeError(f"invalid baseline name: {name!r}")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hstu", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=parse_named_directory, action="append", default=[],
        metavar="NAME=PATH", help="optional row-aligned baseline artifact directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    directories = {"hstu": args.hstu, "transformer": args.transformer}
    for name, directory in args.baseline:
        if name in directories:
            raise RuntimeError(f"duplicate run name: {name}")
        directories[name] = directory
    manifests = {name: verify_manifest(directory) for name, directory in directories.items()}
    predictions = {name: load_predictions(directory) for name, directory in directories.items()}
    for run_name, arrays in predictions.items():
        for key in ALIGNMENT_KEYS:
            if not np.array_equal(predictions["hstu"][key], arrays[key]):
                raise RuntimeError(f"row alignment mismatch for {key}: {run_name}")

    rows = {name: row_metrics(value) for name, value in predictions.items()}
    prefixes = predictions["hstu"]["prefix_lengths"]
    slice_masks = {
        "history_0_20": prefixes <= 20,
        "history_21_50": (prefixes >= 21) & (prefixes <= 50),
        "history_51_80": (prefixes >= 51) & (prefixes <= 80),
        "history_81_plus": prefixes >= 81,
    }
    result = {
        "status": "pass",
        "row_alignment": {
            "rows": int(len(prefixes)),
            "array_sha256": {
                key: hashlib.sha256(
                    np.ascontiguousarray(predictions["hstu"][key]).tobytes()
                ).hexdigest()
                for key in ALIGNMENT_KEYS
            },
        },
        "manifests": manifests,
        "overall": {name: summarize(value) for name, value in rows.items()},
        "transformer_minus_hstu": {
            "hit_rate_at_10": paired_effect(rows["transformer"][0], rows["hstu"][0]),
            "ndcg_at_10": paired_effect(rows["transformer"][1], rows["hstu"][1]),
            "competition_score": paired_effect(rows["transformer"][2], rows["hstu"][2]),
        },
        "onepiece_vs_baselines": {
            f"{candidate}_minus_{reference}": {
                "hit_rate_at_10": paired_effect(rows[candidate][0], rows[reference][0]),
                "ndcg_at_10": paired_effect(rows[candidate][1], rows[reference][1]),
                "competition_score": paired_effect(rows[candidate][2], rows[reference][2]),
            }
            for candidate in ("hstu", "transformer")
            for reference in directories
            if reference not in {"hstu", "transformer"}
        },
        "slices": {
            slice_name: {
                model_name: summarize(tuple(metric[mask] for metric in model_rows))
                for model_name, model_rows in rows.items()
            }
            for slice_name, mask in slice_masks.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
