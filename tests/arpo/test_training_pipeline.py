"""Verify the estimator reaches actor advantages after padding and reordering."""
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

from agent_system.multi_turn_rollout.utils import adjust_batch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage


class TrainingPipelineTests(unittest.TestCase):
    def test_short_rollout_padding_and_reordering_preserve_precomputed_advantages(self):
        config = NS(trainer=NS(n_gpus_per_node=4, nnodes=1),
                    algorithm=NS(use_kl_in_reward=False),
                    actor_rollout_ref=NS(actor=NS(use_kl_loss=True, ppo_micro_batch_size_per_gpu=8),
                                         ref=NS(log_prob_micro_batch_size_per_gpu=8),
                                         rollout=NS(log_prob_micro_batch_size_per_gpu=8)))
        values = torch.tensor([[1., 2.], [-1., -2.], [3., 4.]])
        batch = DataProto.from_dict(tensors={
            "input_ids": torch.ones(3, 4, dtype=torch.long),
            "responses": torch.tensor([[5, 0]] * 3),
            "attention_mask": torch.tensor([[1, 1, 1, 0]] * 3),
            "response_mask": torch.tensor([[1, 0]] * 3),
            "arpo_advantages": values,
        }, non_tensors={"source_row": np.arange(3)})
        padded = adjust_batch(config, batch)
        self.assertEqual(len(padded), 32)
        padded.reorder(torch.arange(31, -1, -1))
        result = compute_advantage(padded, "arpo")
        expected = values[torch.tensor(result.non_tensor_batch["source_row"])] * result.batch["response_mask"]
        torch.testing.assert_close(result.batch["advantages"], expected)
        torch.testing.assert_close(result.batch["returns"], expected)


if __name__ == "__main__":
    unittest.main()
