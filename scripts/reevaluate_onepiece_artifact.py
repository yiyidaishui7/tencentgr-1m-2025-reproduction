"""Re-evaluate a frozen OnePiece model without repeating model training."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

import onepiece_common as common
import onepiece_evaluation_contract as evaluation_contract
import run_onepiece_formal as runner


def required_file(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"configured file does not exist for {name}: {path}")
    return path


def output_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    return Path(value).expanduser().resolve()


SOURCE_MODEL = required_file("ONEPIECE_SOURCE_MODEL")
OUTPUT = output_path("ONEPIECE_REEVAL_OUTPUT")
STATUS = Path(
    os.environ.get(
        "ONEPIECE_REEVAL_STATUS", OUTPUT.parent / f"{OUTPUT.name}_status.json"
    )
).expanduser().resolve()
SOURCE_MODEL_SHA256 = common.sha256(SOURCE_MODEL)
REEVALUATOR_SHA256 = common.sha256(Path(__file__).resolve())
REEVALUATION_CONFIG = {
    "source_model_sha256": SOURCE_MODEL_SHA256,
    "reevaluator_sha256": REEVALUATOR_SHA256,
    "runner_config": runner.RUN_CONFIG,
    "cold_candidate_filtering": True,
    "history_filtering": True,
    "beam_ann_fallback": runner.ENABLE_BEAM_EVAL,
}
REEVALUATION_SIGNATURE = hashlib.sha256(
    json.dumps(
        REEVALUATION_CONFIG, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
).hexdigest()


def write_status(state: str, **values) -> None:
    payload = {"state": state, "time": time.time(), **values}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, STATUS)
    os.chmod(STATUS, 0o600)


def main() -> None:
    started = time.time()
    os.umask(0o077)
    if OUTPUT.exists():
        raise RuntimeError(f"reevaluation output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True, mode=0o700)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    if runner.INDEXER_SHA256 != common.INDEXER_SHA256:
        raise RuntimeError("unexpected indexer hash")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible CUDA device")
    common.set_seed(runner.SEED)
    torch.set_float32_matmul_precision("high")
    runner.write_status = write_status

    with common.INDEXER.open("rb") as handle:
        indexer = pickle.load(handle)
    with np.load(common.CONTRACT, allow_pickle=False) as archive:
        frozen = {name: archive[name] for name in archive.files}
    user_count = len(indexer["u"])
    item_count = len(indexer["i"])

    os.environ["HF_HOME"] = str(common.HOME)
    os.environ["HF_DATASETS_CACHE"] = str(common.HOME / "datasets")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    write_status("preparing", stage="arrow")
    sequences = load_dataset(
        "TAAC2025/TencentGR-1M", "seq", split="train",
        cache_dir=str(common.HOME / "datasets"),
    )
    item_dataset = load_dataset(
        "TAAC2025/TencentGR-1M", "item_feat", split="train",
        cache_dir=str(common.HOME / "datasets"),
    )
    user_dataset = load_dataset(
        "TAAC2025/TencentGR-1M", "user_feat", split="train",
        cache_dir=str(common.HOME / "datasets"),
    )
    candidate_dataset = load_dataset(
        "TAAC2025/TencentGR-1M", "candidate", split="train",
        cache_dir=str(common.HOME / "datasets"),
    )
    if (
        len(sequences) != user_count
        or len(item_dataset) != item_count
        or len(candidate_dataset) != 660000
    ):
        raise RuntimeError("dataset/indexer count mismatch")

    write_status("preparing", stage="features")
    item_features, _ = common.load_dense_features(
        item_dataset, "item_id", runner.ITEM_FEATURES, item_count
    )
    user_features, _ = common.load_dense_features(
        user_dataset, "user_id", runner.USER_FEATURES, user_count
    )
    feature_statistics = {
        name: len(indexer["f"][name])
        for name in runner.ITEM_FEATURES + runner.USER_FEATURES
    }
    feature_types = {
        "user_sparse": runner.USER_FEATURES,
        "user_array": [],
        "user_continual": [],
        "item_sparse": runner.ITEM_FEATURES,
        "context_item_sparse": [],
        "item_array": [],
        "item_continual": [],
        "item_emb": [],
    }

    split_generator = torch.Generator().manual_seed(runner.SPLIT_SEED)
    _, validation_subset = torch.utils.data.random_split(
        range(user_count), [0.9, 0.1], generator=split_generator
    )
    validation_users = {int(value) + 1 for value in validation_subset.indices}
    if not set(map(int, frozen["eval_user_reids"])).issubset(validation_users):
        raise RuntimeError("evaluation contract does not align with validation split")
    eval_dataset = runner.FormalEvalDataset(
        frozen["eval_user_reids"], frozen["eval_target_retrieval_ids"],
        frozen["eval_prefix_lengths"], item_features, user_features,
    )
    write_status("preparing", stage="sequences", processed=0)
    for row_index, row in enumerate(sequences, 1):
        eval_dataset.maybe_fill(row)
        if row_index % 100000 == 0:
            write_status("preparing", stage="sequences", processed=row_index)
    eval_dataset.finalize()
    (
        internal_ids, retrieval_ids, candidate_features,
        cold_start_candidate_count,
    ) = runner.load_candidate_arrays(
        candidate_dataset, indexer["i"],
        frozen["candidate_raw_creative_ids"], feature_statistics,
    )
    evaluation_contract.validate_targets_in_candidate_pool(
        frozen["eval_target_retrieval_ids"], retrieval_ids
    )

    sid_by_item = None
    if runner.ENABLE_SID:
        sid_by_item = np.load(runner.SID_PATH, mmap_mode="r", allow_pickle=False)
        if sid_by_item.shape != (item_count + 1, 2) or sid_by_item.dtype != np.uint16:
            raise RuntimeError("SID mapping shape or dtype mismatch")
        if bool(np.any(sid_by_item[0] != 0)):
            raise RuntimeError("SID padding row must be zero")
        if (
            int(sid_by_item[1:].min()) < 1
            or int(sid_by_item[1:].max()) > runner.SID_CODEBOOK_SIZE
        ):
            raise RuntimeError("SID mapping falls outside the configured codebook")

    sys.path.insert(0, str(common.SOURCE_DIR))
    from model import BaselineModel

    args = common.make_args()
    args.maxlen = runner.MAXLEN
    args.hidden_units = runner.HIDDEN_UNITS
    args.num_blocks = runner.NUM_BLOCKS
    args.num_heads = runner.NUM_HEADS
    args.dropout_rate = 0.1
    args.log_interval = 1_000_000
    args.use_hstu = runner.ARCHITECTURE == "hstu"
    args.sid = runner.ENABLE_SID
    args.sid_codebook_size = runner.SID_CODEBOOK_SIZE
    model = BaselineModel(
        user_count, item_count, feature_statistics, feature_types, args
    )
    checkpoint = torch.load(SOURCE_MODEL, map_location="cpu", weights_only=False)
    source_result = checkpoint.get("result")
    if not isinstance(source_result, dict):
        raise RuntimeError("source model is missing its result receipt")
    source_model = source_result.get("model", {})
    observed = (
        source_model.get("architecture"),
        int(source_model.get("num_blocks", -1)),
        int(source_model.get("hidden_units", -1)),
        int(source_model.get("num_heads", -1)),
        bool(source_model.get("sid_auxiliary", False)),
    )
    expected = (
        "HSTU" if args.use_hstu else "Transformer",
        runner.NUM_BLOCKS,
        runner.HIDDEN_UNITS,
        runner.NUM_HEADS,
        runner.ENABLE_SID,
    )
    if observed != expected:
        raise RuntimeError(f"source model configuration mismatch: {observed} != {expected}")
    model.load_state_dict(checkpoint["state_dict"])
    del checkpoint
    model = model.to(args.device)

    (
        overall, slices, predictions, evaluation_seconds,
        beam_result, evaluation_audit,
    ) = runner.evaluate(
        model, eval_dataset, internal_ids, retrieval_ids, candidate_features,
        item_count, sid_by_item,
    )
    result = {
        "status": "pass",
        "experiment_id": runner.EXPERIMENT_ID,
        "reevaluation_signature": REEVALUATION_SIGNATURE,
        "source_model_sha256": SOURCE_MODEL_SHA256,
        "source_run_signature": source_result.get("run_signature"),
        "metrics": overall,
        "slices": slices,
        "beam_search": beam_result,
        "evaluation_seconds": evaluation_seconds,
        "model": source_model,
        "audit": {
            "eligible_eval_users": len(eval_dataset),
            "official_candidate_rows": 660000,
            "cold_start_candidates": cold_start_candidate_count,
            **evaluation_audit,
        },
        "protocol": {
            "contract_sha256": runner.CONTRACT_SHA256,
            "indexer_sha256": runner.INDEXER_SHA256,
            "cold_candidate_filtering": True,
            "history_filtering": True,
            "beam_ann_fallback": runner.ENABLE_BEAM_EVAL,
            "evaluation_contract_sha256": runner.EVALUATION_CONTRACT_SHA256,
            "upstream_commit": runner.source_contract.EXPECTED_UPSTREAM_COMMIT,
            "upstream_source_sha256": runner.SOURCE_PROVENANCE,
            "runtime_patch_sha256": (
                runner.source_contract.EXPECTED_RUNTIME_PATCH_SHA256
            ),
            "sid_mapping_sha256": runner.SID_SHA256,
        },
        "resources": {
            "device": torch.cuda.get_device_name(0),
            "physical_gpu": runner.PHYSICAL_GPU,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "wall_seconds": time.time() - started,
        },
    }
    metrics_path = OUTPUT / "offline_metrics.json"
    metrics_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    predictions_path = OUTPUT / "offline_predictions.npz"
    np.savez_compressed(predictions_path, **predictions)
    config_path = OUTPUT / "reevaluation_config.json"
    config_path.write_text(
        json.dumps(REEVALUATION_CONFIG, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path = OUTPUT / "source_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "source_model": str(SOURCE_MODEL),
                "source_model_sha256": SOURCE_MODEL_SHA256,
                "source_metrics": source_result.get("metrics"),
                "source_protocol": source_result.get("protocol"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    names = [
        "offline_metrics.json", "offline_predictions.npz",
        "reevaluation_config.json", "source_receipt.json",
    ]
    (OUTPUT / "SHA256SUMS").write_text(
        "".join(f"{common.sha256(OUTPUT / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    for path in OUTPUT.iterdir():
        os.chmod(path, 0o600)
    write_status("complete", **result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status(
            "failed",
            experiment_id=runner.EXPERIMENT_ID,
            reevaluation_signature=REEVALUATION_SIGNATURE,
            error=repr(error),
        )
        raise
