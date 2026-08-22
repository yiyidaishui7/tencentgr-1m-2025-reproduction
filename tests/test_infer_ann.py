import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
if TORCH_AVAILABLE:
    from infer import (
        _read_fbin,
        _read_u64bin,
        _write_topk_ids,
        exact_inner_product_topk,
    )
    from dataset import save_emb


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is not installed in the local test environment')
class ExactInnerProductSearchTests(unittest.TestCase):
    def test_returns_candidate_ids_in_descending_score_order(self) -> None:
        queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        candidates = np.array([[0.9, 0.1], [0.1, 0.8], [0.5, 0.5]], dtype=np.float32)
        candidate_ids = np.array([101, 202, 303], dtype=np.uint64)

        result = exact_inner_product_topk(
            queries,
            candidates,
            candidate_ids,
            device='cpu',
            top_k=2,
            batch_size=1,
        )

        np.testing.assert_array_equal(result, np.array([[101, 303], [202, 303]], dtype=np.uint64))

    def test_binary_round_trip(self) -> None:
        vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        ids = np.array([[11, 12], [21, 22]], dtype=np.uint64)

        with tempfile.TemporaryDirectory() as tmpdir:
            vector_path = Path(tmpdir) / 'vectors.fbin'
            id_path = Path(tmpdir) / 'ids.u64bin'
            save_emb(vectors, vector_path)
            _write_topk_ids(id_path, ids)

            np.testing.assert_array_equal(_read_fbin(vector_path), vectors)
            with self.assertRaises(ValueError):
                _read_u64bin(id_path)

            valid_id_path = Path(tmpdir) / 'candidate_ids.u64bin'
            with open(valid_id_path, 'wb') as file_obj:
                file_obj.write(struct.pack('II', 2, 1))
                np.array([101, 202], dtype=np.uint64).tofile(file_obj)
            np.testing.assert_array_equal(
                _read_u64bin(valid_id_path),
                np.array([101, 202], dtype=np.uint64),
            )


if __name__ == '__main__':
    unittest.main()
