"""Audit the collision-free OnePiece SID alignment experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_onepiece_architectures import (
    ALIGNMENT_KEYS,
    paired_effect,
    row_metrics,
    sha256,
    summarize,
    verify_manifest,
)


RUN_NAMES = ("hstu_no_sid", "hstu_old_sid", "hstu_sid_002", "hstu_sid_005")
SID_RUNS = ("hstu_sid_002", "hstu_sid_005")


def load_archive(directory: Path, require_beam: bool = False) -> dict[str, np.ndarray]:
    with np.load(directory / "offline_predictions.npz", allow_pickle=False) as archive:
        required = [*ALIGNMENT_KEYS, "topk_retrieval_ids"]
        if require_beam:
            required.extend(
                (
                    "beam_topk_retrieval_ids",
                    "beam_valid_candidate_counts",
                    "beam_target_generated",
                )
            )
        missing = [key for key in required if key not in archive]
        if missing:
            raise RuntimeError(f"missing prediction arrays in {directory}: {missing}")
        return {key: archive[key] for key in required}


def validate_reported_metrics(reported: dict, recomputed: dict, label: str) -> None:
    for key in ("hit_rate_at_10", "ndcg_at_10", "competition_score"):
        if key not in reported or not np.isclose(
            float(reported[key]), float(recomputed[key]), rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"reported metric mismatch for {label}: {key}")


def finite_or_none(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def compare(candidate, reference, candidate_summary, reference_summary) -> dict:
    result = {
        "hit_rate_at_10": paired_effect(candidate[0], reference[0]),
        "ndcg_at_10": paired_effect(candidate[1], reference[1]),
        "competition_score": paired_effect(candidate[2], reference[2]),
    }
    reference_score = float(reference_summary["competition_score"])
    result["relative_competition_score"] = (
        float(candidate_summary["competition_score"]) / reference_score - 1.0
        if reference_score else None
    )
    return result


def audit_mapping(directory: Path) -> dict:
    manifest = directory / "SHA256SUMS"
    summary_path = directory / "summary.json"
    mapping_path = directory / "sid_by_internal_id.npy"
    if not manifest.is_file() or not summary_path.is_file() or not mapping_path.is_file():
        raise RuntimeError("collision-free SID mapping artifact is incomplete")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.strip()] = digest
    for name in ("summary.json", "sid_by_internal_id.npy"):
        if expected.get(name) != sha256(directory / name):
            raise RuntimeError(f"SID mapping SHA-256 mismatch: {name}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mapping = np.load(mapping_path, mmap_mode="r", allow_pickle=False)
    if mapping.ndim != 2 or mapping.shape[1] != 2:
        raise RuntimeError(f"unexpected SID mapping shape: {mapping.shape}")
    if summary.get("status") != "pass":
        raise RuntimeError("SID mapping summary did not pass")
    if int(summary["collision_items"]) != 0:
        raise RuntimeError("SID mapping still contains collisions")
    if int(summary["items"]) != int(summary["unique_sid_pairs"]):
        raise RuntimeError("SID mapping pairs are not unique")
    return {
        "status": "pass",
        "sha256": expected["sid_by_internal_id.npy"],
        "items": int(summary["items"]),
        "unique_sid_pairs": int(summary["unique_sid_pairs"]),
        "collision_items": int(summary["collision_items"]),
        "original_pair_retained": int(summary["original_pair_retained"]),
        "top_k_reassigned": int(summary["top_k_reassigned"]),
        "fallback_reassigned": int(summary["fallback_reassigned"]),
        "coarse_sid_moved": int(summary["coarse_sid_moved"]),
        "algorithm": summary["algorithm"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in RUN_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--sid-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directories = {name: getattr(args, name) for name in RUN_NAMES}

    manifests = {name: verify_manifest(path) for name, path in directories.items()}
    archives = {
        name: load_archive(path, require_beam=name in SID_RUNS)
        for name, path in directories.items()
    }
    reference = archives["hstu_no_sid"]
    for name, archive in archives.items():
        for key in ALIGNMENT_KEYS:
            if not np.array_equal(reference[key], archive[key]):
                raise RuntimeError(f"row alignment mismatch for {key}: {name}")

    full_rows = {name: row_metrics(archive) for name, archive in archives.items()}
    full_summary = {name: summarize(rows) for name, rows in full_rows.items()}
    beam_rows = {
        name: row_metrics(
            {
                "topk_retrieval_ids": archives[name]["beam_topk_retrieval_ids"],
                "target_retrieval_ids": archives[name]["target_retrieval_ids"],
            }
        )
        for name in SID_RUNS
    }
    beam_summary = {name: summarize(rows) for name, rows in beam_rows.items()}

    metadata = {}
    beam_diagnostics = {}
    for name, directory in directories.items():
        reported = json.loads((directory / "offline_metrics.json").read_text(encoding="utf-8"))
        validate_reported_metrics(reported["metrics"], full_summary[name], name)
        history = reported.get("training", {}).get("history", [])
        metadata[name] = {
            "experiment_id": reported.get("experiment_id"),
            "model": reported.get("model"),
            "training": {
                "epochs": reported.get("training", {}).get("epochs"),
                "global_steps": reported.get("training", {}).get("global_steps"),
                "final_epoch": {
                    key: finite_or_none(value) for key, value in (history[-1] if history else {}).items()
                },
                "derived_epoch_seconds": float(sum(float(row["seconds"]) for row in history)),
                "sid_component_telemetry_available": not any(
                    isinstance(row.get("mean_sid1_loss"), float)
                    and not np.isfinite(row["mean_sid1_loss"])
                    for row in history
                ),
            },
            "peak_memory_mib": reported.get("resources", {}).get("peak_memory_mib"),
        }
        if name in SID_RUNS:
            validate_reported_metrics(
                reported["beam_search"]["metrics"], beam_summary[name], f"{name} beam"
            )
            valid_counts = archives[name]["beam_valid_candidate_counts"]
            target_generated = archives[name]["beam_target_generated"]
            beam_diagnostics[name] = {
                "metrics": beam_summary[name],
                "beam_size": int(reported["beam_search"]["beam_size"]),
                "generated_pairs": int(reported["beam_search"]["generated_pairs"]),
                "mean_legal_candidate_count": float(valid_counts.mean()),
                "legal_candidate_coverage_at_10": float((valid_counts >= 10).mean()),
                "target_generation_recall": float(target_generated.mean()),
            }

    comparisons = {}
    for candidate in ("hstu_old_sid", "hstu_sid_002", "hstu_sid_005"):
        key = f"{candidate}_minus_hstu_no_sid"
        comparisons[key] = compare(
            full_rows[candidate], full_rows["hstu_no_sid"],
            full_summary[candidate], full_summary["hstu_no_sid"],
        )
    comparisons["hstu_sid_005_minus_hstu_sid_002"] = compare(
        full_rows["hstu_sid_005"], full_rows["hstu_sid_002"],
        full_summary["hstu_sid_005"], full_summary["hstu_sid_002"],
    )

    result = {
        "status": "pass",
        "protocol": {
            "evaluation": "frozen full-candidate Top-10",
            "row_alignment": {"rows": int(len(reference["user_reids"])), "keys": list(ALIGNMENT_KEYS)},
            "candidates": 660000,
            "seeds": 1,
        },
        "mapping": audit_mapping(args.sid_mapping),
        "manifests": manifests,
        "full_candidate": {"overall": full_summary, "paired_comparisons": comparisons},
        "beam_search": {
            "runs": beam_diagnostics,
            "hstu_sid_005_minus_hstu_sid_002": compare(
                beam_rows["hstu_sid_005"], beam_rows["hstu_sid_002"],
                beam_summary["hstu_sid_005"], beam_summary["hstu_sid_002"],
            ),
        },
        "metadata": metadata,
        "limitations": [
            "Single training seed; paired intervals describe the fixed evaluation population, not seed variance.",
            "The 0.02 run's raw SID1/SID2 component telemetry is unavailable because of a logging-key defect; model, predictions, reported totals, hashes, and row alignment remain verified.",
            "Upstream OnePiece README scores use a different feature/filtering/evaluation chain and are context only, not a strict reproduction percentage.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
