import math
from types import SimpleNamespace
import unittest

import torch

from arpo.core_arpo import branch_probability, normalized_topk_entropy, trajectory_advantages


class ARPOCoreTests(unittest.TestCase):
    def test_branch_probability_matches_upstream_event(self):
        # Integrate the exact upstream clipped random-variable condition.
        for base in (0., .2, .5, 1.):
            for delta in (-2., 0., 2.):
                u = torch.linspace(.000005, .999995, 100000)
                observed = ((u - .2 * delta).clamp(0., 1.) <= base).float().mean().item()
                self.assertAlmostEqual(branch_probability(delta, 0., base, .2), observed, places=4)

    def test_topk_entropy_uses_first_twenty_tokens_and_vocab_normalization(self):
        position = {0: SimpleNamespace(logprob=math.log(.25)),
                    1: SimpleNamespace(logprob=math.log(.75))}
        expected = 20 * (-.25 * math.log(.25) - .75 * math.log(.75)) / math.log(100)
        self.assertAlmostEqual(normalized_topk_entropy([position] * 30, 100), expected)

    def test_unequal_turn_counts_do_not_bias_outcome_statistics(self):
        values = trajectory_advantages(["g"] * 4, ["a", "a", "a", "b"], [0, 0, 0, 10])
        expected = 5 / (torch.tensor([0., 10.]).std() + 1e-6)
        torch.testing.assert_close(values, torch.tensor([-expected, -expected, -expected, expected]))

    def test_equal_outcomes_and_inconsistent_trajectory(self):
        values = trajectory_advantages(["g", "g"], ["a", "b"], [10, 10])
        self.assertTrue(torch.equal(values, torch.zeros(2)))
        with self.assertRaises(ValueError):
            trajectory_advantages(["g", "g"], ["a", "a"], [0, 10])


if __name__ == "__main__":
    unittest.main()
