"""Integrity contract for the frozen and runtime-patched OnePiece source."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping


EXPECTED_UPSTREAM_COMMIT = "73e51021dfafb75382baf9acd6a72ce47e5b705b"
EXPECTED_RUNTIME_PATCH_SHA256 = (
    "a15d0191e1cc167a40264f05a95f9e54b000a6e1b93947a80fa62ec0637ebe1a"
)
EXPECTED_SOURCE_HASHES = {
    "model.py": "155c6ed98c6d933bd56f599f8bade13adb7585ae0c0c4e609c2f5648a885dd27",
    "dataset.py": "15d2ce7f5f52ffd2e0d2c17db457326270cb3c6e846000e333408ebe18c6207b",
    "utils.py": "c7b5396103f6cdec229080f14940b2071a636677d369b3cf835ac669931b719e",
    "deepseek_moe.py": "84d7db0b8e276d18fc7338eeb54bff26129eead4689a8b225a7491eaef62f81c",
}


def canonical_python_sha256(path: Path) -> str:
    """Hash Python source after normalizing line endings and final newline."""

    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    normalized = text.rstrip("\r\n") + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def verify_source_files(
    source_dir: Path,
    *,
    expected_hashes: Mapping[str, str] = EXPECTED_SOURCE_HASHES,
) -> dict[str, str]:
    """Verify every critical source file and return its canonical digest."""

    verified: dict[str, str] = {}
    for relative, expected in expected_hashes.items():
        path = source_dir / relative
        if not path.is_file():
            raise RuntimeError(f"missing frozen OnePiece source file: {relative}")
        actual = canonical_python_sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"source SHA-256 mismatch for {relative}: expected {expected}, got {actual}"
            )
        verified[relative] = actual
    return verified
