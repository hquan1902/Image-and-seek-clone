import unittest
from src.research_metrics import average_precision, evaluate


class MetricsTests(unittest.TestCase):
    def test_ap_denominator_and_precision(self):
        self.assertAlmostEqual(average_precision([8, 2, 3], [2, 3, 4], 3), (1/2 + 2/3)/3)
        self.assertEqual(average_precision([2], [2, 3, 4], 1), 1)

    def test_recall_is_designated_target_not_any_positive(self):
        row = {"id": 0, "reference_img_id": 1, "target_img_id": 3, "gt_img_ids": [2, 3]}
        result = evaluate([row], {"0": [2, 4, 3]}, [1, 2, 3, 4], "circo", [1, 3])
        self.assertEqual(result["mAP@1"], 1)
        self.assertEqual(result["R@1"], 0)
        self.assertEqual(result["R@3"], 1)

    def test_invalid_predictions(self):
        row = {"id": 0, "reference_img_id": 1, "target_img_id": 3, "gt_img_ids": [3]}
        for predictions in ({}, {"0": [1, 2, 3]}, {"0": [2, 2, 3]}, {"0": [2, 8, 3]}, {"0": [2]}):
            with self.assertRaises(ValueError):
                evaluate([row], predictions, [1, 2, 3, 4], "circo", [3])

    def test_group_is_computed_before_truncation(self):
        row = {"pairid": 7, "reference": "r", "target_hard": "t", "img_set": {"members": ["r", "t", "g"]}}
        result = evaluate([row], {"7": ["x", "t", "g"]}, ["r", "t", "g", "x"], "cirr", [1, 3])
        self.assertEqual(result["R@1"], 0)
        self.assertEqual(result["group_R@1"], 1)

    def test_fashioniq_uses_candidate_as_reference(self):
        result = evaluate([{"candidate": "r", "target": "t"}], {"0": ["t", "x"]}, ["r", "t", "x"], "fashioniq", [1])
        self.assertEqual(result["R@1"], 1)

    def test_corrupt_ground_truth_rejected(self):
        for positives in ([], [2, 2], [2], [3, 99], [1, 3]):
            row = {"id": 0, "reference_img_id": 1, "target_img_id": 3, "gt_img_ids": positives}
            with self.assertRaises(ValueError):
                evaluate([row], {"0": [2, 3, 4]}, [1, 2, 3, 4], "circo", [3])


if __name__ == "__main__":
    unittest.main()
