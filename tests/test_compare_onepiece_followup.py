from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.compare_onepiece_followup import compare, main, verify_artifacts


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "compare_onepiece_followup.py"
RANKS = {
    "historical_control": [1, 0, 3, 0],
    "historical_mask": [0, 2, 1, 0],
    "aligned_control": [1, 2, 3, 0],
    "aligned_mask": [0, 1, 1, 4],
    "historical_sid_002": [1, 0, 2, 0],
    "historical_sid_005": [0, 2, 0, 1],
    "aligned_sid_002": [1, 1, 2, 0],
    "aligned_sid_005": [1, 2, 0, 1],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_rows(ranks: list[int]) -> tuple[np.ndarray, ...]:
    hits = np.array([float(rank > 0) for rank in ranks])
    ndcg = np.array([1 / math.log2(rank + 1) if rank else 0 for rank in ranks])
    return hits, ndcg, 0.31 * hits + 0.69 * ndcg


def write_manifest(directory: Path) -> None:
    paths = sorted(path for path in directory.iterdir() if path.name != "SHA256SUMS")
    (directory / "SHA256SUMS").write_text(
        "".join(f"{digest(path)}  {path.name}\n" for path in paths), encoding="utf-8"
    )


@pytest.fixture
def runs(tmp_path: Path) -> dict[str, Path]:
    directories = {name: tmp_path / name for name in RANKS}
    for name, directory in directories.items():
        directory.mkdir()
        targets = np.arange(101, 105, dtype=np.uint64)
        topk = np.tile(np.arange(1, 11, dtype=np.uint64), (4, 1))
        for row, rank in enumerate(RANKS[name]):
            if rank:
                topk[row, rank - 1] = targets[row]
        np.savez_compressed(
            directory / "offline_predictions.npz",
            user_reids=np.arange(4, dtype=np.int32),
            target_retrieval_ids=targets,
            prefix_lengths=np.array([10, 30, 60, 90], dtype=np.int16),
            topk_retrieval_ids=topk,
        )
        rows = metric_rows(RANKS[name])
        variant = name.split("_", 1)[1]
        aligned = name.startswith("aligned_")
        reported = {
            "status": "pass",
            "experiment_id": name,
            "metrics": {
                "evaluated_users": 4,
                "hits": int(rows[0].sum()),
                "hit_rate_at_10": float(rows[0].mean()),
                "ndcg_at_10": float(rows[1].mean()),
                "competition_score": float(rows[2].mean()),
            },
            "model": {
                "architecture": "HSTU", "num_blocks": 8, "hidden_units": 512,
                "num_heads": 8, "maxlen": 101,
                "loss": "sample-bias-corrected InfoNCE" + (" + SID1/SID2 cross-entropy" if "sid_" in name else ""),
                "sid_auxiliary": "sid" in name,
                "sid_loss_weight": 0.02 if "002" in name else 0.05 if "005" in name else None,
                "local_path": str(directory / "private.pt"),
            },
            "protocol": {
                "history_filtering": aligned, "cold_candidate_filtering": aligned,
                "post_cutoff_exposure_mask": variant == "mask",
                "contract_sha256": "a" * 64, "indexer_sha256": "b" * 64,
                "training_seed": 20250825, "split_seed": 2025,
                "private_data_path": str(directory / "private-data"),
            },
            "training": {"history": [{"mean_sid_loss": float("nan")}]},
        }
        if aligned:
            reported["protocol"].update(
                beam_ann_fallback=False, evaluation_contract_sha256="f" * 64,
            )
            reported["audit"] = {
                "official_candidate_rows": 660000, "warm_candidates": 511029,
                "cold_start_candidates": 148971, "ann_history_overlap_count": 0,
                "eligible_eval_users": 4, "history_masked_candidate_pairs": 12,
                "local_path": str(directory / "private-receipt"),
            }
            reported["dataset_receipt_reverified"] = True
            historical = directories[f"historical_{variant}"]
            reported["source_model_sha256"] = digest(historical / "model.pt")
            reported["reevaluation_signature"] = "c" * 64
            (directory / "reevaluation_config.json").write_text('{"local": "private"}')
        else:
            (directory / "model.pt").write_bytes(variant.encode())
            reported["run_signature"] = "d" * 64
        (directory / "offline_metrics.json").write_text(json.dumps(reported), encoding="utf-8")
        write_manifest(directory)
    return directories


def run_cli(runs: dict[str, Path], output: Path, *extra: str) -> subprocess.CompletedProcess:
    args = [sys.executable, str(SCRIPT)]
    for name, directory in runs.items():
        args.extend((f"--{name.replace('_', '-')}", str(directory)))
    return subprocess.run(
        [*args, "--output", str(output), *extra], capture_output=True, text=True, check=False
    )


def update_metrics(directory: Path, mutate) -> None:
    path = directory / "offline_metrics.json"
    reported = json.loads(path.read_text(encoding="utf-8"))
    mutate(reported)
    path.write_text(json.dumps(reported), encoding="utf-8")
    write_manifest(directory)


def test_cli_reports_paired_factorial_and_sid_effects_without_private_paths(runs, tmp_path):
    output = tmp_path / "out" / "comparison.json"
    completed = run_cli(runs, output, "--expected-rows", "4")
    assert completed.returncode == 0, completed.stderr
    raw = output.read_text(encoding="utf-8")
    result = json.loads(raw, parse_constant=lambda value: pytest.fail(f"nonfinite JSON: {value}"))
    assert result["status"] == "pass"
    assert result["row_alignment"]["rows"] == 4
    assert len(result["paired_comparisons"]) == 9
    rows = {name: metric_rows(ranks) for name, ranks in RANKS.items()}
    delta = (
        rows["aligned_mask"][2] - rows["aligned_control"][2]
        - rows["historical_mask"][2] + rows["historical_control"][2]
    )
    effect = result["paired_comparisons"]["training_mask_by_protocol_interaction"]["competition_score"]
    assert effect["mean_delta"] == pytest.approx(delta.mean())
    assert effect["normal_95_low"] == pytest.approx(delta.mean() - 1.96 * delta.std(ddof=1) / 2)
    assert effect["normal_95_high"] == pytest.approx(delta.mean() + 1.96 * delta.std(ddof=1) / 2)
    assert "relative_delta" not in effect
    sid_effect = result["paired_comparisons"]["sid_002_minus_aligned_control"]["ndcg_at_10"]
    assert sid_effect["relative_delta"] == pytest.approx(
        rows["aligned_sid_002"][1].mean() / rows["aligned_control"][1].mean() - 1
    )
    assert result["slices"]["history_21_50"]["overall"]["historical_mask"]["evaluated_users"] == 1
    singleton = result["slices"]["history_21_50"]["paired_comparisons"]["protocol_for_mask"]["competition_score"]
    assert singleton["normal_95_low"] is None
    assert singleton["mean_delta"] > 0
    source = result["runs"]["aligned_mask"]
    assert source["source_model_sha256"] == digest(runs["historical_mask"] / "model.pt")
    assert source["reevaluation_signature"] == "c" * 64
    assert source["manifest"]["sha256"] == digest(runs["aligned_mask"] / "SHA256SUMS")
    assert source["dataset_receipt_reverified"] is True
    assert source["audit"] == {
        "official_candidate_rows": 660000, "warm_candidates": 511029,
        "cold_start_candidates": 148971, "ann_history_overlap_count": 0,
        "eligible_eval_users": 4, "history_masked_candidate_pairs": 12,
    }
    assert "training" not in source
    assert "private" not in raw
    assert str(tmp_path) not in raw
    assert any("training-seed variance" in item for item in result["limitations"])


@pytest.mark.parametrize("filename", ["offline_predictions.npz", "offline_metrics.json", "model.pt"])
def test_rejects_tampered_manifest_artifact(runs, tmp_path, filename):
    with (runs["historical_mask"] / filename).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        compare(runs)


@pytest.mark.parametrize("filename", ["offline_predictions.npz", "offline_metrics.json"])
def test_requires_predictions_and_metrics_manifest_coverage(runs, tmp_path, filename):
    manifest = runs["historical_control"] / "SHA256SUMS"
    manifest.write_text("\n".join(line for line in manifest.read_text().splitlines() if filename not in line))
    with pytest.raises(RuntimeError, match="not covered by manifest"):
        compare(runs)


def test_rejects_missing_manifest_referenced_file(runs, tmp_path):
    (runs["historical_control"] / "model.pt").unlink()
    with pytest.raises(RuntimeError, match="missing manifest artifact"):
        compare(runs)


@pytest.mark.parametrize("entry", ["../outside", "C:/private/model.pt", "/absolute/model.pt"])
def test_rejects_unsafe_manifest_path(runs, tmp_path, entry):
    with (runs["historical_control"] / "SHA256SUMS").open("a") as handle:
        handle.write(f"{'0' * 64}  {entry}\n")
    with pytest.raises(RuntimeError, match="unsafe manifest path"):
        compare(runs)


@pytest.mark.parametrize("key", ["user_reids", "target_retrieval_ids", "prefix_lengths"])
def test_rejects_row_alignment_mismatch(runs, tmp_path, key):
    path = runs["aligned_mask"] / "offline_predictions.npz"
    with np.load(path) as archive:
        arrays = dict(archive)
    arrays[key] = arrays[key][::-1]
    np.savez_compressed(path, **arrays)
    write_manifest(path.parent)
    with pytest.raises(RuntimeError, match="row alignment mismatch"):
        compare(runs)


@pytest.mark.parametrize("key,value,expected", [
    ("source_model_sha256", "0" * 64, "source model mismatch"),
    ("status", "fail", "did not pass"),
])
def test_rejects_invalid_source_metadata(runs, tmp_path, key, value, expected):
    update_metrics(runs["aligned_mask"], lambda data: data.update({key: value}))
    with pytest.raises(RuntimeError, match=expected):
        compare(runs)


def test_rejects_reported_metric_mismatch(runs, tmp_path):
    update_metrics(runs["historical_sid_002"], lambda data: data["metrics"].update(competition_score=0.99))
    with pytest.raises(RuntimeError, match="reported metric mismatch"):
        compare(runs)


def test_rejects_unaligned_evaluation_protocol(runs, tmp_path):
    update_metrics(runs["aligned_sid_005"], lambda data: data["protocol"].update(history_filtering=False))
    with pytest.raises(RuntimeError, match="protocol mismatch"):
        compare(runs)


def test_rejects_unexpected_row_count(runs, tmp_path):
    with pytest.raises(RuntimeError, match="expected 78921 rows"):
        compare(runs, expected_rows=78921)


def test_empty_history_slices_produce_null_values(runs, tmp_path, monkeypatch, capsys):
    for directory in runs.values():
        path = directory / "offline_predictions.npz"
        with np.load(path) as archive:
            arrays = dict(archive)
        arrays["prefix_lengths"] = np.full(4, 10, dtype=np.int16)
        np.savez_compressed(path, **arrays)
        write_manifest(directory)
    output = tmp_path / "out.json"
    args = [str(SCRIPT)]
    for name, directory in runs.items():
        args.extend((f"--{name.replace('_', '-')}", str(directory)))
    monkeypatch.setattr(sys, "argv", [*args, "--output", str(output)])
    main()
    result = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == result
    empty = result["slices"]["history_81_plus"]
    assert empty["overall"]["aligned_mask"]["competition_score"] is None
    assert empty["paired_comparisons"]["protocol_for_control"]["hit_rate_at_10"]["mean_delta"] is None


@pytest.mark.parametrize("label,candidate,reference", [
    ("training_mask_under_historical", "historical_mask", "historical_control"),
    ("training_mask_under_aligned", "aligned_mask", "aligned_control"),
    ("protocol_for_control", "aligned_control", "historical_control"),
    ("protocol_for_mask", "aligned_mask", "historical_mask"),
    ("sid_002_minus_aligned_control", "aligned_sid_002", "aligned_control"),
    ("sid_005_minus_aligned_control", "aligned_sid_005", "aligned_control"),
    ("protocol_for_sid_002", "aligned_sid_002", "historical_sid_002"),
    ("protocol_for_sid_005", "aligned_sid_005", "historical_sid_005"),
])
def test_each_comparison_uses_the_intended_pair(runs, label, candidate, reference):
    result = compare(runs, expected_rows=4)
    for index, metric in enumerate(("hit_rate_at_10", "ndcg_at_10", "competition_score")):
        candidate_rows, reference_rows = metric_rows(RANKS[candidate])[index], metric_rows(RANKS[reference])[index]
        delta = candidate_rows - reference_rows
        effect = result["paired_comparisons"][label][metric]
        assert effect["mean_delta"] == pytest.approx(delta.mean())
        assert effect["normal_95_low"] == pytest.approx(delta.mean() - 1.96 * delta.std(ddof=1) / 2)
        assert effect["normal_95_high"] == pytest.approx(delta.mean() + 1.96 * delta.std(ddof=1) / 2)
        assert effect["relative_delta"] == pytest.approx(delta.mean() / reference_rows.mean())


def test_duplicate_manifest_entries_are_rejected(runs):
    manifest = runs["historical_control"] / "SHA256SUMS"
    with manifest.open("a") as handle:
        handle.write(manifest.read_text().splitlines()[0] + "\n")
    with pytest.raises(RuntimeError, match="duplicate manifest path"):
        verify_artifacts(manifest.parent)


@pytest.mark.parametrize("field,value", [("num_blocks", float("nan")), ("hidden_units", float("inf"))])
def test_nonfinite_selected_metadata_cannot_be_published(runs, tmp_path, field, value):
    update_metrics(runs["historical_control"], lambda data: data["model"].update({field: value}))
    completed = run_cli(runs, tmp_path / "out.json")
    assert completed.returncode != 0
    assert not (tmp_path / "out.json").exists()


def test_source_run_signature_must_match_historical_run(runs):
    update_metrics(runs["aligned_mask"], lambda data: data.update(source_run_signature="e" * 64))
    with pytest.raises(RuntimeError, match="source run signature mismatch"):
        compare(runs)


def test_shared_contract_hash_must_match(runs):
    update_metrics(runs["aligned_mask"], lambda data: data["protocol"].update(contract_sha256="e" * 64))
    with pytest.raises(RuntimeError, match="shared protocol mismatch"):
        compare(runs)


def test_sid_weight_must_match_its_named_run(runs):
    update_metrics(runs["historical_sid_002"], lambda data: data["model"].update(sid_loss_weight=0.05))
    with pytest.raises(RuntimeError, match="SID model mismatch"):
        compare(runs)


@pytest.mark.parametrize("key,value", [
    ("beam_ann_fallback", True), ("evaluation_contract_sha256", "e" * 64),
])
def test_aligned_runs_require_identical_beam_disabled_protocol(runs, key, value):
    update_metrics(runs["aligned_mask"], lambda data: data["protocol"].update({key: value}))
    with pytest.raises(RuntimeError, match="protocol"):
        compare(runs)


@pytest.mark.parametrize("field,value", [
    ("upstream_commit", "C:/private/project"),
    ("upstream_source_sha256", {"dataset.py": "C:/private/dataset.py"}),
])
def test_paths_inside_allowed_protocol_fields_are_rejected(runs, field, value):
    update_metrics(runs["historical_mask"], lambda data: data["protocol"].update({field: value}))
    with pytest.raises(RuntimeError, match="invalid protocol"):
        compare(runs)


@pytest.mark.parametrize("field,value", [("architecture", "Transformer"), ("sid_auxiliary", True)])
def test_control_and_mask_must_be_hstu_without_sid(runs, field, value):
    update_metrics(runs["historical_control"], lambda data: data["model"].update({field: value}))
    with pytest.raises(RuntimeError, match="model mismatch"):
        compare(runs)


def test_factorial_models_require_same_dimensions(runs):
    update_metrics(runs["historical_mask"], lambda data: data["model"].update(hidden_units=256))
    with pytest.raises(RuntimeError, match="shared model mismatch"):
        compare(runs)


def test_legacy_control_without_sid_fields_is_accepted(runs):
    def omit_legacy_fields(data):
        data["model"].pop("sid_auxiliary")
        data["model"].pop("sid_loss_weight")
    update_metrics(runs["historical_control"], omit_legacy_fields)
    assert compare(runs)["status"] == "pass"


def test_real_sid_loss_description_is_accepted_without_changing_control_loss(runs):
    for name, directory in runs.items():
        loss = "sample-bias-corrected InfoNCE"
        if "sid_" in name:
            loss += " + SID1/SID2 cross-entropy"
        update_metrics(directory, lambda data, loss=loss: data["model"].update(loss=loss))
    assert compare(runs)["status"] == "pass"


@pytest.mark.parametrize("field", [
    "official_candidate_rows", "warm_candidates", "cold_start_candidates",
    "ann_history_overlap_count", "eligible_eval_users", "history_masked_candidate_pairs",
])
def test_aligned_audit_requires_all_selected_counts(runs, field):
    update_metrics(runs["aligned_mask"], lambda data: data["audit"].pop(field))
    with pytest.raises(RuntimeError, match="aligned audit"):
        compare(runs)


@pytest.mark.parametrize("field,value", [
    ("official_candidate_rows", 659999), ("warm_candidates", 511030),
    ("cold_start_candidates", 148970), ("ann_history_overlap_count", 1),
    ("eligible_eval_users", 78921), ("history_masked_candidate_pairs", -1),
    ("ann_history_overlap_count", False), ("warm_candidates", 511029.0),
    ("history_masked_candidate_pairs", "12"),
])
def test_aligned_audit_rejects_wrong_counts_and_noninteger_values(runs, field, value):
    update_metrics(runs["aligned_sid_002"], lambda data: data["audit"].update({field: value}))
    with pytest.raises(RuntimeError, match="aligned audit"):
        compare(runs)


def test_aligned_history_mask_counts_must_match_across_all_runs(runs):
    update_metrics(runs["aligned_sid_005"], lambda data: data["audit"].update(history_masked_candidate_pairs=13))
    with pytest.raises(RuntimeError, match="shared aligned audit"):
        compare(runs)


@pytest.mark.parametrize("value", [False, None, 1, "true"])
def test_aligned_dataset_receipt_requires_literal_true(runs, value):
    update_metrics(runs["aligned_control"], lambda data: data.update(dataset_receipt_reverified=value))
    with pytest.raises(RuntimeError, match="dataset receipt"):
        compare(runs)


@pytest.mark.parametrize("field", ["audit", "dataset_receipt_reverified"])
def test_aligned_audit_and_dataset_receipt_cannot_be_omitted(runs, field):
    update_metrics(runs["aligned_control"], lambda data: data.pop(field))
    with pytest.raises(RuntimeError, match="aligned audit|dataset receipt"):
        compare(runs)
