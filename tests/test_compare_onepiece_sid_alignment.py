from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).parents[1]


def metrics(topk: np.ndarray, targets: np.ndarray) -> dict:
    matches = topk == targets[:, None]
    hits = matches.any(axis=1)
    ranks = np.where(hits, matches.argmax(axis=1) + 1, 0)
    ndcg = np.zeros(len(targets), dtype=np.float64)
    ndcg[hits] = 1.0 / np.log2(ranks[hits] + 1)
    hr = float(hits.mean())
    ndcg_mean = float(ndcg.mean())
    return {
        "evaluated_users": len(targets),
        "hits": int(hits.sum()),
        "hit_rate_at_10": hr,
        "ndcg_at_10": ndcg_mean,
        "competition_score": 0.31 * hr + 0.69 * ndcg_mean,
    }


def make_run(path: Path, hits: int, beam_hits: int | None = None) -> None:
    path.mkdir()
    targets = np.arange(1, 5, dtype=np.uint64)
    topk = np.full((4, 10), 999, dtype=np.uint64)
    topk[:hits, 0] = targets[:hits]
    arrays = {
        "topk_retrieval_ids": topk,
        "target_retrieval_ids": targets,
        "prefix_lengths": np.array([10, 30, 60, 90], dtype=np.int16),
        "user_reids": np.arange(4, dtype=np.int32),
    }
    beam = None
    if beam_hits is not None:
        beam = np.full((4, 10), 999, dtype=np.uint64)
        beam[:beam_hits, 0] = targets[:beam_hits]
        arrays.update(
            beam_topk_retrieval_ids=beam,
            beam_valid_candidate_counts=np.array([20, 20, 5, 20], dtype=np.int32),
            beam_target_generated=np.array([True, False, False, True]),
        )
    archive = path / "offline_predictions.npz"
    np.savez_compressed(archive, **arrays)
    (path / "SHA256SUMS").write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  offline_predictions.npz\n"
    )
    reported = {
        "experiment_id": path.name,
        "metrics": metrics(topk, targets),
        "training": {"epochs": 1, "global_steps": 4, "history": [{"seconds": 1.0}]},
        "resources": {"peak_memory_mib": 1.0},
        "model": {"sid_loss_weight": None},
    }
    if beam is not None:
        reported["beam_search"] = {
            "metrics": metrics(beam, targets), "beam_size": 20, "generated_pairs": 384
        }
    (path / "offline_metrics.json").write_text(json.dumps(reported))


def make_mapping(path: Path) -> None:
    path.mkdir()
    mapping = np.array([[0, 0], [1, 1], [1, 2], [2, 1], [2, 2]], dtype=np.uint16)
    mapping_path = path / "sid_by_internal_id.npy"
    np.save(mapping_path, mapping)
    summary = {
        "status": "pass", "collision_items": 0, "items": 4, "unique_sid_pairs": 4,
        "original_pair_retained": 2, "top_k_reassigned": 1, "fallback_reassigned": 1,
        "coarse_sid_moved": 0, "algorithm": "test",
    }
    summary_path = path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    (path / "SHA256SUMS").write_text(
        f"{hashlib.sha256(mapping_path.read_bytes()).hexdigest()}  sid_by_internal_id.npy\n"
        f"{hashlib.sha256(summary_path.read_bytes()).hexdigest()}  summary.json\n"
    )


def test_sid_alignment_cli_audits_full_and_beam_metrics(tmp_path: Path):
    no_sid, old_sid, sid_002, sid_005 = (
        tmp_path / name for name in ("no_sid", "old_sid", "sid_002", "sid_005")
    )
    make_run(no_sid, 1)
    make_run(old_sid, 0)
    make_run(sid_002, 2, beam_hits=1)
    make_run(sid_005, 3, beam_hits=2)
    mapping = tmp_path / "mapping"
    make_mapping(mapping)
    output = tmp_path / "comparison.json"
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "compare_onepiece_sid_alignment.py"),
            "--hstu-no-sid", str(no_sid), "--hstu-old-sid", str(old_sid),
            "--hstu-sid-002", str(sid_002), "--hstu-sid-005", str(sid_005),
            "--sid-mapping", str(mapping), "--output", str(output),
        ],
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["status"] == "pass"
    assert result["mapping"]["collision_items"] == 0
    assert result["protocol"]["row_alignment"]["rows"] == 4
    assert result["full_candidate"]["overall"]["hstu_sid_005"]["hits"] == 3
    assert result["beam_search"]["runs"]["hstu_sid_002"]["metrics"]["hits"] == 1
    assert result["beam_search"]["hstu_sid_005_minus_hstu_sid_002"]["competition_score"]["mean_delta"] > 0
