from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
CONTRACT_PATH = ROOT / "scripts" / "onepiece_source_contract.py"


def load_contract_module():
    spec = importlib.util.spec_from_file_location("onepiece_source_contract", CONTRACT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_digest(text: str) -> str:
    normalized = text.replace("\r\n", "\n").rstrip("\r\n") + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def test_source_contract_accepts_only_the_expected_canonical_python_source(tmp_path):
    contract = load_contract_module()
    source = tmp_path / "model.py"
    source.write_text("print('frozen')", encoding="utf-8")
    expected = {"model.py": canonical_digest("print('frozen')")}

    assert contract.verify_source_files(tmp_path, expected_hashes=expected) == expected

    source.write_text("print('tampered')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source SHA-256 mismatch"):
        contract.verify_source_files(tmp_path, expected_hashes=expected)


def test_source_contract_freezes_the_declared_onepiece_commit_and_patch():
    contract = load_contract_module()

    assert contract.EXPECTED_UPSTREAM_COMMIT == (
        "73e51021dfafb75382baf9acd6a72ce47e5b705b"
    )
    assert contract.EXPECTED_RUNTIME_PATCH_SHA256 == (
        "a15d0191e1cc167a40264f05a95f9e54b000a6e1b93947a80fa62ec0637ebe1a"
    )
    assert set(contract.EXPECTED_SOURCE_HASHES) == {
        "model.py",
        "dataset.py",
        "utils.py",
        "deepseek_moe.py",
    }


def test_public_runner_signs_verified_upstream_source_into_run_signature():
    source = (ROOT / "scripts" / "run_onepiece_formal.py").read_text(encoding="utf-8")

    assert "verify_source_files" in source
    assert '"upstream_commit"' in source
    assert '"upstream_source_sha256"' in source
    assert '"runtime_patch_sha256"' in source
