import unittest
import torch
from src.research_fusion import dispersion_normalized_scores, reciprocal_rank_fusion_scores


class FusionTests(unittest.TestCase):
    def test_affine_scale_invariance(self):
        image = torch.tensor([1.0, 2.0, 4.0, 8.0])
        text = torch.tensor([3.0, 1.0, 2.0, 7.0])
        expected = dispersion_normalized_scores(image, text)
        actual = dispersion_normalized_scores(7 * image + 11, 0.2 * text - 5)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5))

    def test_invalid_values_rejected(self):
        with self.assertRaises(ValueError):
            dispersion_normalized_scores(torch.tensor([1.0, float("nan")]), torch.ones(2))

    def test_rrf_is_monotonic_transform_invariant(self):
        image = torch.tensor([1.0, 2.0, 4.0, 8.0])
        text = torch.tensor([3.0, 1.0, 2.0, 7.0])
        expected = reciprocal_rank_fusion_scores(image, text)
        actual = reciprocal_rank_fusion_scores(image.exp(), 3 * text + 2)
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
