"""CPU smoke test of the real trainer dispatch and actor update methods.

Load method ASTs to avoid importing GPU-only FlashAttention/ALFWorld modules.
The forward pass is a tiny differentiable policy; no weights are downloaded.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from omegaconf import OmegaConf

from verl import DataProto
from tapo.core_tapo import compute_tapo_loss, sample_mean


def load_function(path, name, namespace):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class TrainingPathTest(unittest.TestCase):
    def test_advantage_dispatch_and_optimizer_update(self):
        from enum import Enum

        class Estimator(str, Enum):
            GAE = 'gae'
            GRPO = 'grpo'
            GRPO_PASSK = 'grpo_passk'
            REINFORCE_PLUS_PLUS_BASELINE = 'reinforce_plus_plus_baseline'
            REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
            REMAX = 'remax'
            RLOO = 'rloo'
            TAPO = 'tapo'

        compute_advantage = load_function('verl/trainer/ppo/ray_trainer.py', 'compute_advantage',
                                          dict(DataProto=DataProto, AdvantageEstimator=Estimator))
        data = DataProto.from_dict(tensors={
            'responses': torch.ones(4, 2, dtype=torch.long),
            'input_ids': torch.ones(4, 3, dtype=torch.long),
            'attention_mask': torch.ones(4, 3),
            'position_ids': torch.arange(3).expand(4, -1),
            'old_log_probs': torch.zeros(4, 2),
            'ref_log_prob': torch.full((4, 2), -.693147),
            'response_mask': torch.ones(4, 2),
            'tapo_advantage': torch.tensor([1., -1., 1., -1.]),
        })
        data.meta_info['temperature'] = 1.0
        data = compute_advantage(data, 'tapo')
        torch.testing.assert_close(data.batch['advantages'][:, 0], data.batch['tapo_advantage'])

        def append_to_dict(metrics, values):
            for key, value in values.items():
                metrics.setdefault(key, []).append(value)

        update = load_function('verl/workers/actor/dp_actor.py', 'update_policy', dict(
            DataProto=DataProto, torch=torch, compute_tapo_loss=compute_tapo_loss,
            sample_mean=sample_mean, append_to_dict=append_to_dict,
            get_torch_device=lambda: SimpleNamespace(current_device=lambda: 'cpu'),
        ))
        config = OmegaConf.create(dict(
            policy_loss=dict(loss_mode='tapo'), use_kl_loss=False, entropy_coeff=0,
            ppo_mini_batch_size=4, ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1, use_dynamic_bsz=False, clip_ratio=.2,
            clip_ratio_low=None, clip_ratio_high=None, loss_agg_mode='token-mean',
        ))
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)

        def forward(micro_batch, temperature, calculate_entropy):
            self.assertTrue(calculate_entropy)
            logp = torch.nn.functional.logsigmoid(model.weight).expand_as(micro_batch['advantages'])
            return torch.ones_like(logp), logp

        def step():
            norm = model.weight.grad.norm()
            optimizer.step()
            return norm

        actor = SimpleNamespace(config=config, actor_module=model, actor_optimizer=optimizer,
                                _forward_micro_batch=forward, _optimizer_step=step)
        metrics = update(actor, data)
        self.assertTrue(torch.isfinite(model.weight).all())
        self.assertGreater(model.weight.item(), 0)  # successful-only CE supplies a nonzero update
        self.assertIn('actor/tapo_selected_fraction', metrics)


if __name__ == '__main__':
    unittest.main()
