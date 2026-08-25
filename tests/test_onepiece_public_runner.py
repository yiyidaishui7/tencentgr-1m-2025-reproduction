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
