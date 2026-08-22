import math
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from comparison_utils import compare_four_variants, per_user_metric_components


def _prediction(topk, targets=None, prefixes=None, users=None):
    topk = np.asarray(topk, dtype=np.uint64)
    rows = topk.shape[0]
    return {
        "topk_retrieval_ids": topk,
        "target_retrieval_ids": np.asarray(
            targets if targets is not None else np.arange(1, rows + 1), dtype=np.uint64
        ),
        "prefix_lengths": np.asarray(
            prefixes if prefixes is not None else np.arange(rows), dtype=np.int16
        ),
        "user_reids": np.asarray(users if users is not None else np.arange(rows), dtype=np.int32),
    }


class PerUserMetricComponentsTests(unittest.TestCase):
    def test_computes_hit_ndcg_and_weighted_score_per_row(self):
        components = per_user_metric_components(
            np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]]),
            np.array([10, 60, 999]),
            k=3,
        )

        expected_ndcg = np.array([1.0, 1.0 / math.log2(4), 0.0])
        np.testing.assert_array_equal(components["hit"], np.array([1.0, 1.0, 0.0]))
        np.testing.assert_allclose(components["ndcg"], expected_ndcg)
        np.testing.assert_allclose(
            components["score"], 0.31 * components["hit"] + 0.69 * expected_ndcg
        )


class FourVariantComparisonTests(unittest.TestCase):
    def setUp(self):
        self.targets = [1, 2, 3, 4]
        self.predictions = {
            "mm101": _prediction([[1, 9], [9, 8], [9, 3], [9, 8]], self.targets),
            "nomm101": _prediction([[1, 9], [2, 9], [9, 8], [9, 8]], self.targets),
            "mm50": _prediction([[1, 9], [9, 8], [9, 8], [9, 4]], self.targets),
            "nomm50": _prediction([[1, 9], [2, 9], [3, 9], [4, 9]], self.targets),
        }

    def test_reports_alignment_aggregate_pairwise_and_interaction(self):
        result = compare_four_variants(self.predictions, k=2)

        self.assertEqual(result["alignment"]["evaluated_users"], 4)
        self.assertTrue(result["alignment"]["all_equal"])
        self.assertEqual(result["aggregate"]["nomm50"]["hits"], 4)
        self.assertIn("no_mm_effect_at_101", result["paired"])
        self.assertIn("window_effect_with_mm", result["paired"])
        self.assertIn("window_effect_without_mm", result["paired"])
        self.assertAlmostEqual(result["interaction"]["mean_score_delta"], 0.5)

    def test_rejects_row_misalignment(self):
        self.predictions["nomm50"] = _prediction(
            [[1, 9], [2, 9], [3, 9], [4, 9]], self.targets, users=[0, 1, 2, 99]
        )

        with self.assertRaisesRegex(ValueError, "user_reids"):
            compare_four_variants(self.predictions, k=2)

    def test_requires_all_four_named_variants(self):
        del self.predictions["nomm50"]

        with self.assertRaisesRegex(ValueError, "exactly"):
            compare_four_variants(self.predictions, k=2)


class ComparisonCliTests(unittest.TestCase):
    def test_writes_machine_readable_four_way_report(self):
        targets = [1, 2]
        variants = {
            "mm101": _prediction([[1, 9], [9, 8]], targets),
            "nomm101": _prediction([[1, 9], [2, 9]], targets),
            "mm50": _prediction([[1, 9], [9, 2]], targets),
            "nomm50": _prediction([[1, 9], [2, 9]], targets),
        }
        script = Path(__file__).parents[1] / "scripts" / "compare_four_variants.py"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            arguments = [sys.executable, str(script)]
            for name, arrays in variants.items():
                archive = root / f"{name}.npz"
                np.savez_compressed(archive, **arrays)
                arguments.extend([f"--{name}", str(archive)])
            output = root / "comparison.json"
            arguments.extend(["--output", str(output), "--k", "2"])

            completed = subprocess.run(arguments, text=True, capture_output=True, check=False)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["alignment"]["evaluated_users"], 2)
            self.assertIn("interaction", report)


if __name__ == "__main__":
    unittest.main()
