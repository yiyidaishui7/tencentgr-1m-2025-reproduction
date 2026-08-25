from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_protocol_builder_cli_freezes_aligned_rows(tmp_path):
    arrays = {
        "user_reids": np.array([3, 9], dtype=np.int32),
        "target_retrieval_ids": np.array([0, 2], dtype=np.uint64),
        "prefix_lengths": np.array([7, 81], dtype=np.int16),
    }
    predictions = []
    for index in range(2):
        path = tmp_path / f"predictions_{index}.npz"
        np.savez_compressed(path, **arrays)
        predictions.append(path)
    retrieval_map = tmp_path / "retrieval.json"
    retrieval_map.write_text(
        json.dumps({"0": 100, "1": 101, "2": 102}), encoding="utf-8"
    )
    output = tmp_path / "protocol"
    script = Path(__file__).parents[1] / "scripts" / "build_onepiece_protocol.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--predictions",
            *(str(path) for path in predictions),
            "--retrieval-map",
            str(retrieval_map),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metadata = json.loads((output / "protocol_metadata.json").read_text(encoding="utf-8"))
    assert metadata["eligible_eval_users"] == 2
    assert metadata["candidate_count"] == 3
    with np.load(output / "offline_contract.npz", allow_pickle=False) as contract:
        np.testing.assert_array_equal(contract["eval_target_raw_creative_ids"], [100, 102])

