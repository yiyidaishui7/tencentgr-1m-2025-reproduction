"""Runtime helpers shared by training and inference entry points."""

from __future__ import annotations

import importlib
import random
from pathlib import Path
from typing import Any

import numpy as np


def _load_torch_npu() -> bool:
    """Register the Ascend backend when ``torch_npu`` is installed."""
    try:
        importlib.import_module("torch_npu")
    except ImportError:
        return False
    return True


def resolve_device(requested: str | None, torch_module: Any) -> str:
    """Resolve ``auto``/CUDA/Ascend/CPU into a validated PyTorch device.

    ``torch_npu`` is imported lazily so the repository remains importable on
    ordinary CPU and CUDA machines.
    """
    device = (requested or "auto").strip().lower()

    if device == "auto":
        if _load_torch_npu() and getattr(torch_module, "npu", None):
            if torch_module.npu.is_available():
                device = "npu:0"
        if device == "auto" and torch_module.cuda.is_available():
            device = "cuda:0"
        if device == "auto":
            device = "cpu"

    if device.startswith("npu"):
        if not _load_torch_npu():
            raise RuntimeError(
                "Ascend device requested, but torch_npu is not installed. "
                "Use the torch_npu build matching the remote CANN and PyTorch versions."
            )
        npu = getattr(torch_module, "npu", None)
        if npu is None or not npu.is_available():
            raise RuntimeError("Ascend device requested, but torch.npu is unavailable.")
        npu.set_device(device)
        return device

    if device.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda is unavailable.")

    if not (device == "cpu" or device.startswith("cuda")):
        raise ValueError(f"Unsupported device: {requested!r}")

    return device


def uses_accelerator(device: str) -> bool:
    """Return whether asynchronous accelerator transfers may be useful."""
    return device.startswith(("cuda", "npu"))


def load_model_state_dict(path: str | Path, device: str, torch_module: Any):
    """Load model-only weights from SafeTensors or a tensor-only PyTorch file.

    SafeTensors is preferred for published checkpoints. Legacy ``.pt`` state
    dicts are loaded with ``weights_only=True`` so arbitrary pickle objects are
    not deserialized.
    """

    checkpoint = Path(path).expanduser()
    if checkpoint.suffix.lower() == ".safetensors":
        try:
            safetensors_torch = importlib.import_module("safetensors.torch")
        except ImportError as exc:
            raise RuntimeError(
                "Loading .safetensors weights requires the safetensors package."
            ) from exc
        return safetensors_torch.load_file(str(checkpoint), device=device)

    return torch_module.load(
        checkpoint,
        map_location=torch_module.device(device),
        weights_only=True,
    )


def seed_everything(seed: int, torch_module: Any) -> None:
    """Seed Python, NumPy, and PyTorch and disable cuDNN autotuning."""

    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)

    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.manual_seed_all(seed)

    cudnn = getattr(getattr(torch_module, "backends", None), "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False


def active_mm_emb_ids(mm_emb_ids, disabled: bool) -> list[str]:
    """Return a detached list of enabled multimodal feature IDs."""
    return [] if disabled else list(mm_emb_ids)
