import torch

from early_edit_reward_distillation.trajectory import deterministic_rollout


def test_trajectory_module_imports_without_pipeline():
    assert callable(deterministic_rollout)

