import unittest

import torch

from tempo.core_tempo import trajectory_prefix_advantages


def upstream_oracle(responses, masks, scores, groups, normalize=True):
    """Small direct translation of the upstream prefix filtering formula."""
    result = torch.zeros_like(responses, dtype=torch.float32)
    for row, group in enumerate(groups):
        previous = [i for i, value in enumerate(groups) if value == group]
        outcomes = scores[previous]
        scale = outcomes.std() + 1e-6 if normalize else 1.
        for token in range(responses.shape[1]):
            if not masks[row, token]:
                break
            branch = [i for i in previous if responses[i, token] == responses[row, token]]
            result[row, token] = (scores[row] - outcomes.mean()
                                  + scores[branch].mean() - scores[previous].mean()) / scale
            previous = branch
    return result * masks


class TEMPOCoreTests(unittest.TestCase):
    def test_matches_upstream_on_random_single_turn_groups(self):
        generator = torch.Generator().manual_seed(7)
        for normalize in (True, False):
            for _ in range(20):
                responses = torch.randint(0, 3, (8, 9), generator=generator)
                scores = torch.randint(0, 2, (8,), generator=generator).float() * 10
                groups = ["a"] * 4 + ["b"] * 4
                masks = torch.ones_like(responses)
                expected = upstream_oracle(responses, masks, scores, groups, normalize)
                actual = trajectory_prefix_advantages(responses, masks, groups, list(range(8)),
                                                       [0] * 8, scores, normalize)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_multiturn_prefixes_survive_row_reordering(self):
        # Both trajectories share the first action, then diverge at the next turn.
        responses = torch.tensor([[1, 2], [3, 0], [1, 2], [4, 0]])
        masks = torch.tensor([[1, 1], [1, 0], [1, 1], [1, 0]])
        groups, ids, steps, rewards = ["g"] * 4, ["a", "a", "b", "b"], [0, 1, 0, 1], [0, 0, 10, 10]
        values = trajectory_prefix_advantages(responses, masks, groups, ids, steps, rewards, normalize=False)
        torch.testing.assert_close(values, torch.tensor([[-5., -5.], [-10., 0.], [5., 5.], [10., 0.]]))
        order = [3, 0, 2, 1]
        shuffled = trajectory_prefix_advantages(responses[order], masks[order], groups,
                                                [ids[i] for i in order], [steps[i] for i in order],
                                                [rewards[i] for i in order], normalize=False)
        torch.testing.assert_close(shuffled, values[order])

    def test_singletons_equal_outcomes_padding_and_invalid_duplicates(self):
        tokens = torch.tensor([[1, 2, 99], [1, 3, 88]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
        for ids, groups, scores in ((["a", "b"], ["g", "g"], [10, 10]),
                                    (["a", "b"], ["g", "h"], [10, 0])):
            values = trajectory_prefix_advantages(tokens, mask, groups, ids, [0, 0], scores)
            self.assertTrue(torch.isfinite(values).all())
            self.assertTrue(torch.equal(values, torch.zeros_like(values)))
        with self.assertRaises(ValueError):
            trajectory_prefix_advantages(tokens, mask, ["g"] * 2, ["a"] * 2, [0, 0], [0, 0])

    def test_turn_boundaries_cannot_merge_different_action_sequences(self):
        # [1] then [2,3] and [1,2] then [3] are different action histories.
        responses = torch.tensor([[1, 0], [2, 3], [1, 2], [3, 0]])
        masks = torch.tensor([[1, 0], [1, 1], [1, 1], [1, 0]])
        values = trajectory_prefix_advantages(responses, masks, ["g"] * 4, ["a", "a", "b", "b"],
                                             [0, 1, 0, 1], [0, 0, 10, 10], normalize=False)
        self.assertEqual(values[1, 0].item(), -5.)
        self.assertEqual(values[3, 0].item(), 5.)


if __name__ == "__main__":
    unittest.main()
