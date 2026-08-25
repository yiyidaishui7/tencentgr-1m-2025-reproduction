"""Compare row-aligned OnePiece capacity-scaling runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_onepiece_architectures import (
    ALIGNMENT_KEYS,
    load_predictions,
    paired_effect,
    row_metrics,
    summarize,
    verify_manifest,
)


def named_directory(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("run name and path must be non-empty")
    return name, Path(raw_path)


def read_metadata(directory: Path, recomputed: dict[str, float | int]) -> dict:
    path = directory / "offline_metrics.json"
    if not path.is_file():
        return {"reported_metrics": None}
    result = json.loads(path.read_text(encoding="utf-8"))
    reported = result.get("metrics", {})
    for key in ("hit_rate_at_10", "ndcg_at_10", "competition_score"):
        if key not in reported or not np.isclose(
            float(reported[key]), float(recomputed[key]), rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"reported metric mismatch for {directory}: {key}")
    return {
        "experiment_id": result.get("experiment_id"),
        "run_signature": result.get("run_signature"),
        "model": result.get("model"),
        "training": result.get("training"),
        "resources": result.get("resources"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=named_directory, action="append", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    directories: dict[str, Path] = {}
    for name, directory in args.run:
        if name in directories:
            raise RuntimeError(f"duplicate run name: {name}")
        directories[name] = directory
    if len(directories) < 2:
        raise RuntimeError("at least two runs are required")
    if args.reference not in directories:
        raise RuntimeError(f"reference run is missing: {args.reference}")

    manifests = {name: verify_manifest(path) for name, path in directories.items()}
    predictions = {name: load_predictions(path) for name, path in directories.items()}
    reference_predictions = predictions[args.reference]
    for run_name, arrays in predictions.items():
        for key in ALIGNMENT_KEYS:
            if not np.array_equal(reference_predictions[key], arrays[key]):
                raise RuntimeError(f"row alignment mismatch for {key}: {run_name}")

    rows = {name: row_metrics(value) for name, value in predictions.items()}
    overall = {name: summarize(value) for name, value in rows.items()}
    metadata = {
        name: read_metadata(directories[name], overall[name]) for name in directories
    }
    reference_rows = rows[args.reference]
    comparisons = {
        f"{name}_minus_{args.reference}": {
            "hit_rate_at_10": paired_effect(value[0], reference_rows[0]),
            "ndcg_at_10": paired_effect(value[1], reference_rows[1]),
            "competition_score": paired_effect(value[2], reference_rows[2]),
            "relative_competition_score": (
                (overall[name]["competition_score"] / overall[args.reference]["competition_score"]) - 1.0
                if overall[args.reference]["competition_score"] else None
            ),
        }
        for name, value in rows.items()
        if name != args.reference
    }
    prefixes = reference_predictions["prefix_lengths"]
    masks = {
        "history_0_20": prefixes <= 20,
        "history_21_50": (prefixes >= 21) & (prefixes <= 50),
        "history_51_80": (prefixes >= 51) & (prefixes <= 80),
        "history_81_plus": prefixes >= 81,
    }
    result = {
        "status": "pass",
        "reference": args.reference,
        "row_alignment": {"rows": int(len(prefixes)), "keys": list(ALIGNMENT_KEYS)},
        "manifests": manifests,
        "overall": overall,
        "metadata": metadata,
        "comparisons": comparisons,
        "slices": {
            slice_name: {
                name: summarize(tuple(metric[mask] for metric in run_rows))
                for name, run_rows in rows.items()
            }
            for slice_name, mask in masks.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
