from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).parents[1]


def make_run(path: Path, hits: int, users: np.ndarray | None = None) -> None:
    targets = np.arange(1, 5, dtype=np.uint64)
    topk = np.full((4, 10), 999, dtype=np.uint64)
    topk[:hits, 0] = targets[:hits]
    archive = path / "offline_predictions.npz"
    path.mkdir()
    np.savez_compressed(
        archive,
        topk_retrieval_ids=topk,
        target_retrieval_ids=targets,
        prefix_lengths=np.array([10, 30, 60, 90], dtype=np.int16),
        user_reids=users if users is not None else np.arange(4, dtype=np.int32),
    )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (path / "SHA256SUMS").write_text(f"{digest}  offline_predictions.npz\n")
    hr = hits / 4
    metrics = {
        "experiment_id": path.name,
        "metrics": {
            "hit_rate_at_10": hr,
            "ndcg_at_10": hr,
            "competition_score": hr,
        },
        "model": {"parameters": 100 + hits},
    }
    (path / "offline_metrics.json").write_text(json.dumps(metrics))


def test_scaling_cli_reports_row_aligned_effects(tmp_path: Path):
    base, medium, large = (tmp_path / name for name in ("base", "medium", "large"))
    make_run(base, 1)
    make_run(medium, 2)
    make_run(large, 3)
    output = tmp_path / "comparison.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_onepiece_scaling.py"),
            "--run", f"hstu_4x128={base}",
            "--run", f"hstu_8x256={medium}",
            "--run", f"hstu_8x512={large}",
            "--reference", "hstu_4x128",
            "--output", str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["row_alignment"]["rows"] == 4
    assert result["overall"]["hstu_8x512"]["competition_score"] == 0.75
    assert result["comparisons"]["hstu_8x512_minus_hstu_4x128"]["relative_competition_score"] == 2.0
    assert result["slices"]["history_21_50"]["hstu_8x256"]["hits"] == 1


def test_scaling_cli_rejects_misaligned_rows(tmp_path: Path):
    base, bad = tmp_path / "base", tmp_path / "bad"
    make_run(base, 1)
    make_run(bad, 2, users=np.array([0, 1, 2, 99], dtype=np.int32))
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_onepiece_scaling.py"),
            "--run", f"base={base}",
            "--run", f"bad={bad}",
            "--reference", "base",
            "--output", str(tmp_path / "comparison.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "user_reids" in completed.stderr
