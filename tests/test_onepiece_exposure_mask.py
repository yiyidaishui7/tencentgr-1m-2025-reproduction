from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
CONTRACT_PATH = ROOT / "scripts" / "onepiece_training_contract.py"


def load_contract_module():
    spec = importlib.util.spec_from_file_location(
        "onepiece_training_contract", CONTRACT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_post_cutoff_exposure_mask_matches_frozen_onepiece_boundary():
    contract = load_contract_module()
    cutoff = contract.POST_CUTOFF_EXPOSURE_TIMESTAMP

    assert cutoff == 1_748_620_800
    assert contract.ranking_loss_weight(None, enabled=True) == 1.0
    assert contract.ranking_loss_weight(cutoff - 1, enabled=True) == 1.0
    assert contract.ranking_loss_weight(cutoff, enabled=True) == 1.0
    assert contract.ranking_loss_weight(cutoff + 1, enabled=True) == 0.0


def test_disabled_exposure_mask_preserves_the_frozen_control_protocol():
    contract = load_contract_module()

    assert contract.ranking_loss_weight(
        contract.POST_CUTOFF_EXPOSURE_TIMESTAMP + 1, enabled=False
    ) == 1.0


def test_public_runner_signs_the_mask_setting_into_its_run_contract():
    source = (ROOT / "scripts" / "run_onepiece_formal.py").read_text(
        encoding="utf-8"
    )

    assert "ONEPIECE_ENABLE_POST_CUTOFF_EXPOSURE_MASK" in source
    assert "ONEPIECE_POST_CUTOFF_EXPOSURE_TIMESTAMP" in source
    assert '"post_cutoff_exposure_mask"' in source
    assert "ranking_loss_weight" in source
