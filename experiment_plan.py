"""Build a drift-resistant plan for the fixed-seed 2x2 reproduction matrix."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


REQUIRED_VARIANTS: dict[str, tuple[int, bool]] = {
    "mm101": (101, True),
    "nomm101": (101, False),
    "mm50": (50, True),
    "nomm50": (50, False),
}


def _path_text(value: str | Path) -> str:
    """Return a command-friendly path without making it machine absolute."""

    return str(value).replace("\\", "/")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _validate_matrix(config: Mapping[str, Any]) -> None:
    controlled = _require_mapping(config.get("controlled_experiment"), "controlled_experiment")
    factors = _require_mapping(controlled.get("factors"), "controlled_experiment.factors")
    variants = _require_mapping(controlled.get("variants"), "controlled_experiment.variants")

    if set(variants) != set(REQUIRED_VARIANTS):
        raise ValueError(
            "controlled_experiment.variants must contain exactly "
            f"{', '.join(REQUIRED_VARIANTS)}"
        )
    if set(factors.get("max_sequence_length", ())) != {50, 101}:
        raise ValueError("max_sequence_length factors must contain exactly 101 and 50")
    if set(factors.get("multimodal_enabled", ())) != {False, True}:
        raise ValueError("multimodal_enabled factors must contain exactly true and false")


def build_experiment_plan(
    config: Mapping[str, Any],
    *,
    data_path: str | Path,
    runs_root: str | Path,
    eval_root: str | Path,
    scratch_root: str | Path,
    device: str,
    python_executable: str = "python",
) -> dict[str, Any]:
    """Return exact training/evaluation argv for the four controlled variants.

    The returned object is deliberately data-only. It can be reviewed, stored,
    or consumed by an external scheduler without this helper starting a GPU job.
    """

    _validate_matrix(config)
    model = _require_mapping(config.get("model"), "model")
    training = _require_mapping(config.get("training"), "training")

    data = _path_text(data_path)
    runs = _path_text(runs_root).rstrip("/")
    evaluations = _path_text(eval_root).rstrip("/")
    scratch = _path_text(scratch_root).rstrip("/")
    seed = int(training["seed"])
    valid_ratio = float(training["validation_ratio"])
    batch_size = int(training["batch_size"])
    learning_rate = float(training["learning_rate"])
    epochs = int(training["epochs"])
    hidden_units = int(model["hidden_units"])
    num_blocks = int(model["num_blocks"])
    num_heads = int(model["num_heads"])
    dropout_rate = float(model["dropout_rate"])
    mm_fields = [str(field) for field in model["multimodal_embedding_fields"]]

    variants: list[dict[str, Any]] = []
    for name, (maxlen, multimodal_enabled) in REQUIRED_VARIANTS.items():
        run_dir = f"{runs}/{name}"
        checkpoint_root = f"{run_dir}/checkpoints"
        eval_dir = f"{evaluations}/{name}"
        scratch_dir = f"{scratch}/{name}"
        train_argv = [
            python_executable,
            "main.py",
            "--data_path",
            data,
            "--output_dir",
            run_dir,
            "--device",
            device,
            "--seed",
            str(seed),
            "--valid_ratio",
            str(valid_ratio),
            "--maxlen",
            str(maxlen),
            "--batch_size",
            str(batch_size),
            "--lr",
            str(learning_rate),
            "--num_epochs",
            str(epochs),
            "--hidden_units",
            str(hidden_units),
            "--num_blocks",
            str(num_blocks),
            "--num_heads",
            str(num_heads),
            "--dropout_rate",
            str(dropout_rate),
            "--mm_emb_id",
            *mm_fields,
        ]
        evaluate_argv = [
            python_executable,
            "offline_eval.py",
            "--data_path",
            data,
            "--checkpoint",
            "{checkpoint}",
            "--output_dir",
            eval_dir,
            "--scratch_dir",
            scratch_dir,
            "--device",
            device,
            "--seed",
            str(seed),
            "--valid_ratio",
            str(valid_ratio),
            "--maxlen",
            str(maxlen),
            "--hidden_units",
            str(hidden_units),
            "--num_blocks",
            str(num_blocks),
            "--num_heads",
            str(num_heads),
            "--dropout_rate",
            str(dropout_rate),
            "--mm_emb_id",
            *mm_fields,
        ]
        if not multimodal_enabled:
            train_argv.append("--disable_mm_emb")
            evaluate_argv.append("--disable_mm_emb")

        variants.append(
            {
                "name": name,
                "maxlen": maxlen,
                "multimodal_enabled": multimodal_enabled,
                "train": {
                    "argv": train_argv,
                    "env": {
                        "TRAIN_CKPT_PATH": checkpoint_root,
                        "TRAIN_LOG_PATH": f"{run_dir}/logs",
                        "TRAIN_TF_EVENTS_PATH": f"{run_dir}/events",
                    },
                },
                "evaluate": {
                    "argv": evaluate_argv,
                    "checkpoint_glob": f"{checkpoint_root}/global_step*.valid_loss=*/model.pt",
                    "predictions": f"{eval_dir}/offline_predictions.npz",
                    "metrics": f"{eval_dir}/offline_metrics.json",
                },
            }
        )

    comparison_argv = [python_executable, "scripts/compare_four_variants.py"]
    for variant in variants:
        comparison_argv.extend(
            [f"--{variant['name']}", variant["evaluate"]["predictions"]]
        )
    comparison_argv.extend(["--output", f"{evaluations}/four_way_comparison.json"])

    return {
        "schema_version": 1,
        "contract": {
            "seed": seed,
            "valid_ratio": valid_ratio,
            "candidate_pool": config["evaluation"]["candidate_pool"],
            "row_alignment_required": True,
            "execution": "review-only plan; commands are not started by the planner",
        },
        "variants": variants,
        "comparison": {"argv": comparison_argv},
    }

