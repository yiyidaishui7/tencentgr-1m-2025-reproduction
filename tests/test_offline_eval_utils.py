import math
import unittest

import numpy as np

from offline_eval_utils import competition_metrics, select_last_click_holdout


class HoldoutSelectionTests(unittest.TestCase):
    def test_selects_last_click_and_excludes_it_from_history(self) -> None:
        item_ids = np.array([11, 22, 33, 44, 55], dtype=np.int64)
        action_types = np.array([0, 1, 0, 1, 0], dtype=np.int8)

        holdout = select_last_click_holdout(item_ids, action_types)

        self.assertIsNotNone(holdout)
        self.assertEqual(holdout.target_position, 3)
        self.assertEqual(holdout.target_item_id, 44)
        self.assertEqual(holdout.prefix_length, 3)

    def test_returns_none_when_sequence_has_no_valid_click(self) -> None:
        item_ids = np.array([11, 0, 33], dtype=np.int64)
        action_types = np.array([0, 1, 0], dtype=np.int8)

        self.assertIsNone(select_last_click_holdout(item_ids, action_types))

    def test_rejects_mismatched_event_arrays(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            select_last_click_holdout(
                np.array([1, 2], dtype=np.int64),
                np.array([1], dtype=np.int8),
            )


class CompetitionMetricTests(unittest.TestCase):
    def test_computes_official_top10_weighting(self) -> None:
        topk_ids = np.array(
            [
                [10, 20, 30],
                [40, 50, 60],
                [70, 80, 90],
            ],
            dtype=np.uint64,
        )
        target_ids = np.array([10, 60, 999], dtype=np.uint64)

        metrics = competition_metrics(topk_ids, target_ids, k=3)

        self.assertEqual(metrics["evaluated_users"], 3)
        self.assertEqual(metrics["hits"], 2)
        self.assertAlmostEqual(metrics["hit_rate_at_10"], 2 / 3)
        self.assertAlmostEqual(metrics["ndcg_at_10"], (1.0 + 1.0 / math.log2(4)) / 3)
        self.assertAlmostEqual(
            metrics["competition_score"],
            0.31 * metrics["hit_rate_at_10"] + 0.69 * metrics["ndcg_at_10"],
        )

    def test_rejects_empty_or_misaligned_predictions(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one user"):
            competition_metrics(np.empty((0, 10)), np.empty((0,)), k=10)

        with self.assertRaisesRegex(ValueError, "target count"):
            competition_metrics(np.ones((2, 10)), np.ones((1,)), k=10)

        with self.assertRaisesRegex(ValueError, "between 1"):
            competition_metrics(np.ones((2, 3)), np.ones((2,)), k=4)


if __name__ == "__main__":
    unittest.main()
