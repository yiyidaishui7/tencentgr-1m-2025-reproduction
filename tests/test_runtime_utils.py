from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from runtime_utils import (
    active_mm_emb_ids,
    load_model_state_dict,
    resolve_device,
    seed_everything,
    uses_accelerator,
)


class _Backend:
    def __init__(self, available: bool) -> None:
        self._available = available
        self.selected: str | None = None

    def is_available(self) -> bool:
        return self._available

    def set_device(self, device: str) -> None:
        self.selected = device


class RuntimeUtilsTest(unittest.TestCase):
    def test_seed_everything_configures_reproducible_torch_state(self) -> None:
        cuda = _Backend(True)
        cuda.manual_seed_all = Mock()
        torch = SimpleNamespace(
            cuda=cuda,
            manual_seed=Mock(),
            backends=SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True)),
        )

        seed_everything(2025, torch)

        torch.manual_seed.assert_called_once_with(2025)
        cuda.manual_seed_all.assert_called_once_with(2025)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_multimodal_ids_can_be_disabled_without_mutating_input(self) -> None:
        source = ['81', '82']
        self.assertEqual(active_mm_emb_ids(source, disabled=False), ['81', '82'])
        self.assertEqual(active_mm_emb_ids(source, disabled=True), [])
        self.assertEqual(source, ['81', '82'])

    def test_auto_falls_back_to_cpu(self) -> None:
        torch = SimpleNamespace(cuda=_Backend(False))
        with patch("runtime_utils._load_torch_npu", return_value=False):
            self.assertEqual(resolve_device("auto", torch), "cpu")

    def test_auto_selects_cuda_when_available(self) -> None:
        torch = SimpleNamespace(cuda=_Backend(True))
        with patch("runtime_utils._load_torch_npu", return_value=False):
            self.assertEqual(resolve_device("auto", torch), "cuda:0")

    def test_auto_prefers_npu_and_sets_it(self) -> None:
        npu = _Backend(True)
        torch = SimpleNamespace(cuda=_Backend(True), npu=npu)
        with patch("runtime_utils._load_torch_npu", return_value=True):
            self.assertEqual(resolve_device("auto", torch), "npu:0")
        self.assertEqual(npu.selected, "npu:0")

    def test_requested_npu_requires_torch_npu(self) -> None:
        torch = SimpleNamespace(cuda=_Backend(False))
        with patch("runtime_utils._load_torch_npu", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "torch_npu is not installed"):
                resolve_device("npu:0", torch)

    def test_accelerator_detection(self) -> None:
        self.assertTrue(uses_accelerator("npu:0"))
        self.assertTrue(uses_accelerator("cuda:1"))
        self.assertFalse(uses_accelerator("cpu"))

    def test_pytorch_state_dict_uses_weights_only_loading(self) -> None:
        torch = SimpleNamespace(load=Mock(return_value={"weight": "tensor"}), device=Mock())

        state = load_model_state_dict("model.pt", "cpu", torch)

        self.assertEqual(state, {"weight": "tensor"})
        torch.device.assert_called_once_with("cpu")
        torch.load.assert_called_once_with(
            __import__("pathlib").Path("model.pt"),
            map_location=torch.device.return_value,
            weights_only=True,
        )

    def test_safetensors_state_dict_uses_safe_loader(self) -> None:
        safe_loader = SimpleNamespace(load_file=Mock(return_value={"weight": "tensor"}))
        torch = SimpleNamespace(load=Mock(), device=Mock())
        with patch("runtime_utils.importlib.import_module", return_value=safe_loader) as importer:
            state = load_model_state_dict("model.safetensors", "cuda:0", torch)

        self.assertEqual(state, {"weight": "tensor"})
        importer.assert_called_once_with("safetensors.torch")
        safe_loader.load_file.assert_called_once_with("model.safetensors", device="cuda:0")
        torch.load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
