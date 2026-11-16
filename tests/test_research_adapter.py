import unittest
try:
    import torch
    import torch.nn.functional as F
    from src.research_adapter import (AdaptiveComposer, MultiDomainResidualComposer,
                                      multi_positive_nce, full_gallery_cross_entropy,
                                      hard_negative_margin_loss, group_cross_entropy)
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Needs PyTorch")
class AdapterTests(unittest.TestCase):
    def test_initialized_as_static_fusion(self):
        torch.manual_seed(7)
        model = AdaptiveComposer(8, 16, 0.2, use_residual=True, gate_input="full")
        r, t = F.normalize(torch.randn(4, 8), dim=-1), F.normalize(torch.randn(4, 8), dim=-1)
        q, w = model(r, t)
        torch.testing.assert_close(q, F.normalize(0.2*r+0.8*t, dim=-1))
        torch.testing.assert_close(w, torch.tensor([0.2, 0.8]).expand(4, 2))

    def test_duplicate_targets_not_false_negatives(self):
        q = torch.eye(2)
        self.assertAlmostEqual(multi_positive_nce(q, q, torch.tensor([1, 1])).item(), 0)
        loss = multi_positive_nce(q, q.flip(0), torch.tensor([1, 2]))
        self.assertGreater(loss.item(), 1)

    def test_backward_is_finite(self):
        model = AdaptiveComposer(8, 16)
        q, _ = model(torch.randn(4, 8), torch.randn(4, 8))
        loss = multi_positive_nce(q, torch.randn(4, 8), torch.arange(4))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_full_gallery_loss_prefers_correct_target(self):
        gallery = torch.eye(4)
        targets = torch.tensor([1, 3])
        references = torch.tensor([0, 2])
        good_loss = full_gallery_cross_entropy(gallery[targets], gallery, targets, references)
        bad_loss = full_gallery_cross_entropy(gallery[references], gallery, targets, references)
        self.assertLess(good_loss.item(), bad_loss.item())
        smoothed = full_gallery_cross_entropy(
            gallery[targets], gallery, targets, references, label_smoothing=0.05)
        self.assertTrue(torch.isfinite(smoothed))

    def test_multidomain_residual_starts_at_static_baseline(self):
        torch.manual_seed(11)
        model = MultiDomainResidualComposer(8, hidden=4, dropout=0.2, residual_scale=0.1)
        reference, text = torch.randn(3, 8), torch.randn(3, 8)
        query, residual = model(reference, text, 0.3)
        expected = F.normalize(0.3*F.normalize(reference, dim=-1)
                               + 0.7*F.normalize(text, dim=-1), dim=-1)
        torch.testing.assert_close(query, expected)
        torch.testing.assert_close(residual, torch.zeros_like(residual))

    def test_hard_negative_loss_prefers_separated_positive(self):
        gallery = F.normalize(torch.tensor([[1., 0.], [0.9, 0.1], [0., 1.]]), dim=-1)
        targets = torch.tensor([0])
        close = hard_negative_margin_loss(gallery[0:1], gallery, targets, topk=1)
        separated = hard_negative_margin_loss(torch.tensor([[1., -1.]]), gallery, targets, topk=1)
        self.assertLess(separated.item(), close.item())

    def test_group_loss_prefers_target_over_group_distractors(self):
        gallery = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        groups = torch.tensor([[0, 1, 2]])
        targets = torch.tensor([0])
        correct = group_cross_entropy(torch.tensor([[1., 0.]]), gallery, groups, targets)
        wrong = group_cross_entropy(torch.tensor([[0., 1.]]), gallery, groups, targets)
        self.assertLess(correct.item(), wrong.item())


if __name__ == "__main__":
    unittest.main()
