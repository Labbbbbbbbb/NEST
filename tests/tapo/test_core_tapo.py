import unittest

import torch

from tapo.core_tapo import compute_tapo_loss, trajectory_advantages


class TAPOTests(unittest.TestCase):
    def test_trajectory_weighting_and_reordering(self):
        # A long failed trajectory must count once, just like a short success.
        result = trajectory_advantages(['q'] * 4, ['a', 'b', 'a', 'a'], [0, 1, 0, 0])
        torch.testing.assert_close(result, torch.tensor([-1., 1., -1., -1.]))

    def test_homogeneous_and_singleton(self):
        result = trajectory_advantages(['q', 'q', 'r'], ['a', 'b', 'c'], [1, 1, 0])
        torch.testing.assert_close(result, torch.zeros(3))

    def test_invalid_labels(self):
        with self.assertRaises(ValueError):
            trajectory_advantages(['q'], ['a'], [.5])
        with self.assertRaises(ValueError):
            trajectory_advantages(['q', 'q'], ['a', 'a'], [0, 1])

    def test_clipping_and_sign(self):
        # Positive ratio > 1.2 clips; negative ratio > 1.2 remains penalized.
        logp = torch.tensor([[.4, .4]], requires_grad=True)
        loss, _ = compute_tapo_loss(logp, torch.zeros_like(logp),
                                   torch.tensor([[1., -1.]]), torch.ones_like(logp),
                                   torch.ones_like(logp))
        loss.backward()
        self.assertEqual(logp.grad[0, 0].item(), 0)
        self.assertGreater(logp.grad[0, 1].item(), 0)

    def test_entropy_padding_and_empty_backward(self):
        logp = torch.zeros((2, 3), requires_grad=True)
        mask = torch.tensor([[1, 1, 0], [0, 0, 0]])
        entropy = torch.tensor([[.01, .02, 10.], [10., 10., 10.]])
        loss, stats = compute_tapo_loss(logp, logp.detach(), torch.ones_like(logp),
                                        mask, entropy)
        loss.backward()
        self.assertEqual(stats['actor/tapo_selected_fraction'], .5)
        self.assertEqual(logp.grad[0, 0].item(), 0)
        self.assertLess(logp.grad[0, 1].item(), 0)
        self.assertEqual(logp.grad[0, 2].item(), 0)
        logp.grad.zero_()
        zero, _ = compute_tapo_loss(logp, logp.detach(), torch.ones_like(logp),
                                   mask, torch.zeros_like(logp))
        zero.backward()
        self.assertEqual(zero.item(), 0)
        self.assertEqual(logp.grad.abs().sum().item(), 0)

    def test_mean_of_valid_sample_means(self):
        logp = torch.zeros((3, 2), requires_grad=True)
        advantage = torch.tensor([[1., 1.], [-1., -1.], [5., 5.]])
        mask = torch.tensor([[1, 0], [1, 1], [0, 0]])
        loss, _ = compute_tapo_loss(logp, logp.detach(), advantage, mask,
                                   torch.ones_like(logp))
        self.assertAlmostEqual(loss.item(), 0)


if __name__ == '__main__':
    unittest.main()
