"""Verify eight OnePiece artifacts and compare training and evaluation effects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

import numpy as np

try:
    from .compare_onepiece_architectures import (
        ALIGNMENT_KEYS, load_predictions, paired_effect, row_metrics, sha256, summarize,
    )
except ImportError:  # Direct script invocation.
    from compare_onepiece_architectures import (
        ALIGNMENT_KEYS, load_predictions, paired_effect, row_metrics, sha256, summarize,
    )


RUN_NAMES = (
    "historical_control", "historical_mask", "aligned_control", "aligned_mask",
    "historical_sid_002", "historical_sid_005", "aligned_sid_002", "aligned_sid_005",
)
METRIC_NAMES = ("hit_rate_at_10", "ndcg_at_10", "competition_score")
# Frozen TencentGR-1M candidate protocol; evaluation rows may be a smaller fixture.
ALIGNED_AUDIT_COUNTS = {
    "official_candidate_rows": 660000,
    "warm_candidates": 511029,
    "cold_start_candidates": 148971,
    "ann_history_overlap_count": 0,
}
MODEL_FIELDS = (
    "architecture", "hidden_units", "loss", "maxlen", "num_blocks", "num_heads",
    "parameters", "sid_auxiliary", "sid_codebook_size", "sid_loss_delay_steps",
    "sid_loss_warmup_steps", "sid_loss_weight",
)
PROTOCOL_FIELDS = (
    "beam_eval", "beam_ann_fallback", "cold_candidate_filtering", "contract_sha256",
    "evaluation_contract_sha256", "history_filtering", "indexer_sha256",
    "post_cutoff_exposure_mask", "post_cutoff_exposure_timestamp",
    "runtime_patch_sha256", "sid_mapping_sha256", "split_seed", "training_seed",
    "upstream_commit",
)
PAIRINGS = {
    "training_mask_under_historical": ("historical_mask", "historical_control"),
    "training_mask_under_aligned": ("aligned_mask", "aligned_control"),
    "protocol_for_control": ("aligned_control", "historical_control"),
    "protocol_for_mask": ("aligned_mask", "historical_mask"),
    "sid_002_minus_aligned_control": ("aligned_sid_002", "aligned_control"),
    "sid_005_minus_aligned_control": ("aligned_sid_005", "aligned_control"),
    "protocol_for_sid_002": ("aligned_sid_002", "historical_sid_002"),
    "protocol_for_sid_005": ("aligned_sid_005", "historical_sid_005"),
}


def verify_artifacts(directory: Path) -> dict:
    """Require every manifest entry to exist, stay in the run, and match its hash."""
    manifest = next(
        (directory / name for name in ("SHA256SUMS", "remote_checksums.sha256")
         if (directory / name).is_file()), None,
    )
    if manifest is None:
        raise RuntimeError(f"missing SHA-256 manifest: {directory.name}")
    files: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]) is None:
            raise RuntimeError(f"invalid SHA-256 manifest entry: {directory.name}")
        digest, name = fields
        name = name.removeprefix("*")
        relative = PurePosixPath(name)
        if (
            not name or "\\" in name or ":" in name or relative.is_absolute()
            or PureWindowsPath(name).drive or ".." in relative.parts
            or not (directory / name).resolve().is_relative_to(directory.resolve())
        ):
            raise RuntimeError(f"unsafe manifest path: {name}")
        if name in files:
            raise RuntimeError(f"duplicate manifest path: {name}")
        files[name] = digest.lower()
    for name in ("offline_predictions.npz", "offline_metrics.json"):
        if name not in files:
            raise RuntimeError(f"{name} not covered by manifest: {directory.name}")
    for name, expected in files.items():
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"missing manifest artifact: {directory.name}/{name}")
        if sha256(path) != expected:
            raise RuntimeError(f"SHA-256 mismatch: {directory.name}/{name}")
    return {"filename": manifest.name, "sha256": sha256(manifest), "files": files}


def validate_arrays(arrays: dict[str, np.ndarray], name: str) -> None:
    rows = len(arrays["user_reids"])
    if rows == 0:
        raise RuntimeError(f"empty prediction population: {name}")
    for key, array in arrays.items():
        shape = (rows, 10) if key == "topk_retrieval_ids" else (rows,)
        if array.shape != shape or not np.issubdtype(array.dtype, np.integer):
            raise RuntimeError(f"invalid prediction shape or integer dtype: {name}/{key}")
    if np.any(arrays["prefix_lengths"] < 0):
        raise RuntimeError(f"negative prefix length: {name}")
    if len(np.unique(arrays["user_reids"])) != rows:
        raise RuntimeError(f"duplicate evaluation users: {name}")


def validate_portable_metadata(model: dict, protocol: dict, name: str) -> None:
    for key in MODEL_FIELDS:
        if key not in model or model[key] is None:
            continue
        value = model[key]
        if key == "loss":
            valid = value == (
                "sample-bias-corrected InfoNCE + SID1/SID2 cross-entropy"
                if "sid_" in name else "sample-bias-corrected InfoNCE"
            )
        elif key == "architecture":
            valid = isinstance(value, str) and not any(mark in value for mark in ("/", "\\", ":"))
        elif key == "sid_auxiliary":
            valid = isinstance(value, bool)
        elif key == "sid_loss_weight":
            valid = type(value) in (int, float) and np.isfinite(value) and value >= 0
        else:
            valid = type(value) is int and value >= 0
        if not valid:
            raise RuntimeError(f"invalid model metadata: {name}/{key}")
    for key in PROTOCOL_FIELDS:
        if key not in protocol:
            continue
        value = protocol[key]
        if key.endswith("sha256") or key == "upstream_commit":
            length = 40 if key == "upstream_commit" else 64
            valid = (key == "sid_mapping_sha256" and value is None) or (
                isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None
            )
        elif key in ("beam_eval", "beam_ann_fallback", "cold_candidate_filtering", "history_filtering", "post_cutoff_exposure_mask"):
            valid = isinstance(value, bool)
        else:
            valid = type(value) is int and value >= 0
        if not valid:
            raise RuntimeError(f"invalid protocol metadata: {name}/{key}")
    for key, value in protocol.get("upstream_source_sha256", {}).items():
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise RuntimeError(f"invalid protocol source hash: {name}/{key}")


def metadata(reported: dict, manifest: dict, recomputed: dict, name: str) -> dict:
    if reported.get("status") != "pass":
        raise RuntimeError(f"reported artifact did not pass: {name}")
    for key, expected in recomputed.items():
        actual = reported.get("metrics", {}).get(key)
        if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
                or not np.isclose(actual, expected, rtol=0.0, atol=1e-12)):
            raise RuntimeError(f"reported metric mismatch: {name}/{key}")
    protocol = reported.get("protocol", {})
    model = reported.get("model", {})
    validate_portable_metadata(model, protocol, name)
    aligned = name.startswith("aligned_")
    if (protocol.get("history_filtering") is not aligned
            or protocol.get("cold_candidate_filtering", False) is not aligned):
        raise RuntimeError(f"evaluation protocol mismatch: {name}")
    if aligned and (
        protocol.get("beam_ann_fallback") is not False or protocol.get("beam_eval", False) is not False
        or reported.get("beam_search") is not None
    ):
        raise RuntimeError(f"Beam-disabled evaluation protocol mismatch: {name}")
    if name.startswith("historical_"):
        if protocol.get("post_cutoff_exposure_mask", False) is not (name == "historical_mask"):
            raise RuntimeError(f"training mask protocol mismatch: {name}")
        if "model.pt" not in manifest["files"]:
            raise RuntimeError(f"model.pt not covered by manifest: {name}")
    if model.get("architecture") != "HSTU":
        raise RuntimeError(f"HSTU model mismatch: {name}")
    # The original control predates explicit SID metadata; its absent flag means False.
    sid_default = False if name == "historical_control" else None
    if name in ("historical_control", "historical_mask") and (
        model.get("sid_auxiliary", sid_default) is not False or model.get("sid_loss_weight") is not None
    ):
        raise RuntimeError(f"no-SID model mismatch: {name}")
    if name.startswith("historical_sid_"):
        weight = 0.02 if name.endswith("002") else 0.05
        if model.get("sid_auxiliary") is not True or model.get("sid_loss_weight") != weight:
            raise RuntimeError(f"SID model mismatch: {name}")
    result = {
        "manifest": manifest,
        "model": {key: model[key] for key in MODEL_FIELDS if key in model},
        "protocol": {key: protocol[key] for key in PROTOCOL_FIELDS if key in protocol},
        "metrics": recomputed,
    }
    if aligned:
        if reported.get("dataset_receipt_reverified") is not True:
            raise RuntimeError(f"dataset receipt was not reverified: {name}")
        audit = reported.get("audit")
        if not isinstance(audit, dict):
            raise RuntimeError(f"missing aligned audit: {name}")
        required = {**ALIGNED_AUDIT_COUNTS, "eligible_eval_users": recomputed["evaluated_users"]}
        for key, expected in required.items():
            if type(audit.get(key)) is not int or audit[key] != expected:
                raise RuntimeError(f"aligned audit count mismatch: {name}/{key}")
        masked_pairs = audit.get("history_masked_candidate_pairs")
        if type(masked_pairs) is not int or masked_pairs < 0:
            raise RuntimeError(f"invalid aligned audit: {name}/history_masked_candidate_pairs")
        result["audit"] = {**required, "history_masked_candidate_pairs": masked_pairs}
        result["dataset_receipt_reverified"] = True
    experiment_id = reported.get("experiment_id")
    if not isinstance(experiment_id, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", experiment_id) is None:
        raise RuntimeError(f"invalid portable experiment_id: {name}")
    result["experiment_id"] = experiment_id
    for key in ("source_model_sha256", "source_run_signature", "run_signature", "reevaluation_signature"):
        if key in reported:
            if not isinstance(reported[key], str) or re.fullmatch(r"[0-9a-f]{64}", reported[key]) is None:
                raise RuntimeError(f"invalid {key}: {name}")
            result[key] = reported[key]
    if not aligned:
        if ("source_model_sha256" in reported
                and reported["source_model_sha256"] != manifest["files"]["model.pt"]):
            raise RuntimeError(f"source model mismatch: {name}")
        result["source_model_sha256"] = manifest["files"]["model.pt"]
    source_hashes = protocol.get("upstream_source_sha256", {})
    result["protocol"]["upstream_source_sha256"] = {
        key: source_hashes[key]
        for key in ("dataset.py", "deepseek_moe.py", "model.py", "utils.py")
        if key in source_hashes
    }
    return result


def effect(candidate: np.ndarray, reference: np.ndarray, *, relative: bool = True) -> dict:
    count = len(candidate)
    if count >= 2:
        result = paired_effect(candidate, reference)
    else:
        result = {
            "mean_delta": float((candidate - reference)[0]) if count else None,
            "normal_95_low": None, "normal_95_high": None,
        }
    if relative:
        denominator = float(reference.mean()) if count else 0.0
        result["relative_delta"] = float((candidate - reference).mean()) / denominator if denominator else None
    return result


def summarize_rows(rows: tuple[np.ndarray, ...]) -> dict:
    if len(rows[0]):
        return summarize(rows)
    return {"evaluated_users": 0, "hits": 0, **{key: None for key in METRIC_NAMES}}


def statistical_report(rows: dict[str, tuple[np.ndarray, ...]]) -> dict:
    comparisons = {
        label: {
            metric: effect(rows[candidate][index], rows[reference][index])
            for index, metric in enumerate(METRIC_NAMES)
        }
        for label, (candidate, reference) in PAIRINGS.items()
    }
    comparisons["training_mask_by_protocol_interaction"] = {
        metric: effect(
            rows["aligned_mask"][index] - rows["aligned_control"][index],
            rows["historical_mask"][index] - rows["historical_control"][index],
            relative=False,
        )
        for index, metric in enumerate(METRIC_NAMES)
    }
    return {
        "overall": {name: summarize_rows(value) for name, value in rows.items()},
        "paired_comparisons": comparisons,
    }


def compare(directories: dict[str, Path], expected_rows: int | None = None) -> dict:
    manifests = {name: verify_artifacts(directories[name]) for name in RUN_NAMES}
    arrays = {name: load_predictions(directories[name]) for name in RUN_NAMES}
    reference = arrays["historical_control"]
    for name, predictions in arrays.items():
        validate_arrays(predictions, name)
        for key in ALIGNMENT_KEYS:
            if not np.array_equal(reference[key], predictions[key]):
                raise RuntimeError(f"row alignment mismatch: {name}/{key}")
    prefixes = reference["prefix_lengths"]
    if expected_rows is not None and len(prefixes) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, got {len(prefixes)}")
    rows = {name: row_metrics(predictions) for name, predictions in arrays.items()}
    sources = {
        name: metadata(
            json.loads((directories[name] / "offline_metrics.json").read_text(encoding="utf-8")),
            manifests[name], summarize_rows(rows[name]), name,
        )
        for name in RUN_NAMES
    }
    for variant in ("control", "mask", "sid_002", "sid_005"):
        historical, aligned = sources[f"historical_{variant}"], sources[f"aligned_{variant}"]
        if aligned.get("source_model_sha256") != historical["source_model_sha256"]:
            raise RuntimeError(f"source model mismatch: aligned_{variant}")
        if ("source_run_signature" in aligned and "run_signature" in historical
                and aligned["source_run_signature"] != historical["run_signature"]):
            raise RuntimeError(f"source run signature mismatch: aligned_{variant}")
    for key in ("contract_sha256", "indexer_sha256"):
        hashes = [source["protocol"].get(key) for source in sources.values()]
        if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes):
            raise RuntimeError(f"missing or invalid protocol hash: {key}")
        if len(set(hashes)) != 1:
            raise RuntimeError(f"shared protocol mismatch: {key}")
    aligned_contracts = {
        source["protocol"].get("evaluation_contract_sha256")
        for name, source in sources.items() if name.startswith("aligned_")
    }
    if None in aligned_contracts or len(aligned_contracts) != 1:
        raise RuntimeError("shared aligned evaluation protocol mismatch")
    aligned_masked_pairs = {
        source["audit"]["history_masked_candidate_pairs"]
        for name, source in sources.items() if name.startswith("aligned_")
    }
    if len(aligned_masked_pairs) != 1:
        raise RuntimeError("shared aligned audit mismatch: history_masked_candidate_pairs")
    for key in ("architecture", "hidden_units", "maxlen", "num_blocks", "num_heads"):
        values = {source["model"].get(key) for source in sources.values()}
        if None in values or len(values) != 1:
            raise RuntimeError(f"shared model mismatch: {key}")
    for key in ("training_seed", "split_seed"):
        values = {
            source["protocol"].get(key)
            for name, source in sources.items() if name.startswith("historical_")
        }
        if None in values or len(values) != 1:
            raise RuntimeError(f"shared training protocol mismatch: {key}")
    slice_masks = {
        "history_0_20": prefixes <= 20,
        "history_21_50": (prefixes >= 21) & (prefixes <= 50),
        "history_51_80": (prefixes >= 51) & (prefixes <= 80),
        "history_81_plus": prefixes >= 81,
    }
    return {
        "schema_version": 1,
        "status": "pass",
        "row_alignment": {
            "rows": len(prefixes), "keys": list(ALIGNMENT_KEYS),
            "array_sha256": {
                key: hashlib.sha256(np.ascontiguousarray(reference[key]).tobytes()).hexdigest()
                for key in ALIGNMENT_KEYS
            },
        },
        "runs": sources,
        **statistical_report(rows),
        "slices": {
            label: statistical_report({
                name: tuple(metric[mask] for metric in values) for name, values in rows.items()
            })
            for label, mask in slice_masks.items()
        },
        "inference": {
            "training_seeds": 1,
            "confidence_interval": "paired per-user normal approximation: mean +/- 1.96 * sample SE",
            "interaction": "(aligned_mask - aligned_control) - (historical_mask - historical_control)",
            "relative_delta": "mean_delta / reference_mean; null when the reference mean is zero",
        },
        "limitations": [
            "Single training seed; paired user intervals do not measure training-seed variance.",
            "Protocol effects reuse the same checkpoint; they measure evaluation and post-processing changes.",
            "Intervals are unadjusted for multiple comparisons; they are null for fewer than two users.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in RUN_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, help="optional production population size, e.g. 78921")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare({name: getattr(args, name) for name in RUN_NAMES}, args.expected_rows)
    serialized = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
