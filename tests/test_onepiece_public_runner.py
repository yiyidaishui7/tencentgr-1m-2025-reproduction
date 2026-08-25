from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_public_runner_contains_no_private_machine_paths():
    files = [
        ROOT / "scripts" / "run_onepiece_formal.py",
        ROOT / "scripts" / "onepiece_common.py",
        ROOT / "docs" / "ONEPIECE_RUNBOOK.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "/tmp/sunche" not in text
    assert "s84448890" not in text
    assert re.search(r"hf_[A-Za-z0-9]{20,}", text) is None


def test_control_configs_only_change_architecture():
    hstu = json.loads((ROOT / "configs" / "onepiece_hstu_4x128.json").read_text())
    transformer = json.loads(
        (ROOT / "configs" / "onepiece_transformer_4x128.json").read_text()
    )
    assert hstu.pop("architecture") == "HSTU"
    assert "Transformer" in transformer.pop("architecture")
    assert hstu == transformer


def test_scaling_configs_keep_the_frozen_training_protocol():
    base = json.loads((ROOT / "configs" / "onepiece_hstu_4x128.json").read_text())
    medium = json.loads((ROOT / "configs" / "onepiece_hstu_8x256.json").read_text())
    large = json.loads((ROOT / "configs" / "onepiece_hstu_8x512.json").read_text())
    variable_fields = {"hidden_units", "num_blocks", "num_heads"}
    for candidate in (medium, large):
        assert candidate["architecture"] == "HSTU"
        assert {
            key: value for key, value in candidate.items() if key not in variable_fields
        } == {
            key: value for key, value in base.items() if key not in variable_fields
        }
    assert (medium["num_blocks"], medium["hidden_units"], medium["num_heads"]) == (8, 256, 8)
    assert (large["num_blocks"], large["hidden_units"], large["num_heads"]) == (8, 512, 8)


def test_public_runner_exposes_safe_scaling_overrides():
    source = (ROOT / "scripts" / "run_onepiece_formal.py").read_text(encoding="utf-8")
    for name in (
        "ONEPIECE_EXPERIMENT_ID",
        "ONEPIECE_HIDDEN_UNITS",
        "ONEPIECE_NUM_BLOCKS",
        "ONEPIECE_NUM_HEADS",
        "RUN_SIGNATURE",
    ):
        assert name in source


def test_common_configuration_requires_path_neutral_environment():
    source = (ROOT / "scripts" / "onepiece_common.py").read_text(encoding="utf-8")
    for name in (
        "ONEPIECE_WORK_ROOT",
        "ONEPIECE_SOURCE_DIR",
        "ONEPIECE_CONTRACT",
        "ONEPIECE_INDEXER",
        "ONEPIECE_INDEXER_SHA256",
    ):
        assert name in source
    assert "hidden_units=128" in source
