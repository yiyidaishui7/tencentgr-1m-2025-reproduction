from __future__ import annotations

import unittest

from candidate_utils import candidate_item_column, canonical_id_key, decode_candidate_feature


class CandidateUtilsTest(unittest.TestCase):
    def test_canonical_id_key_normalizes_integer_ids(self):
        self.assertEqual(canonical_id_key(20000000029), "20000000029")

    def test_public_item_column(self) -> None:
        self.assertEqual(candidate_item_column({"item_id", "retrieval_id"}), "item_id")

    def test_legacy_item_column_takes_precedence(self) -> None:
        self.assertEqual(candidate_item_column({"creative_id", "item_id"}), "creative_id")

    def test_public_warm_feature(self) -> None:
        value = {"cold_start": 0, "feature_value": "42"}
        self.assertEqual(decode_candidate_feature(value), 42)

    def test_public_cold_feature_maps_to_zero(self) -> None:
        value = {"cold_start": 1, "feature_value": "unseen"}
        self.assertEqual(decode_candidate_feature(value), 0)

    def test_legacy_feature(self) -> None:
        self.assertEqual(decode_candidate_feature({"is_str": 0, "feature_value": "7"}), 7)
        self.assertEqual(decode_candidate_feature({"is_str": 1, "feature_value": "x"}), 0)

    def test_scalar_and_null_feature(self) -> None:
        self.assertEqual(decode_candidate_feature(8), 8)
        self.assertEqual(decode_candidate_feature(None), 0)


if __name__ == "__main__":
    unittest.main()
