import math

import torch
from PIL import Image
import pytest

from early_edit_reward_distillation.continuous_strength import (
    CallableRewardScorer,
    ContinuousStrengthConfig,
    RewardUnavailable,
    TrajectoryTrace,
    estimate_edit_token_mask,
    rollout_strengths,
    select_winner,
)


def trace(prompt, values):
    states = [torch.tensor(v, dtype=torch.float32) for v in values]
    velocities = [states[i + 1] - states[i] for i in range(len(states) - 1)]
    return TrajectoryTrace(prompt, states, velocities, [1.0] * len(velocities), [(1.0, 0.5)] * len(velocities), states[-1])


def test_config_defaults_and_validation():
    cfg = ContinuousStrengthConfig()
    assert cfg.num_candidates == 4
    assert cfg.strengths == (0.0, 0.25, 0.5, 0.75, 1.0)
    with pytest.raises(ValueError):
        ContinuousStrengthConfig(num_candidates=2)


def test_same_initial_state_mask_excludes_source_by_construction():
    preservation = trace("p", [[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]])
    edited = trace("e", [[[0, 0], [0, 0], [0, 0]], [[0, 0], [1, 0], [0, 0]]])
    mask, scores = estimate_edit_token_mask(preservation, edited, [0], quantile=0.5, min_ratio=0.0, max_ratio=0.5)
    assert mask.shape == (3,)
    assert mask.dtype == torch.bool
    assert scores.shape == (3,)
    assert int(mask.sum()) == 1


def test_mask_ratio_bounds_and_constant_difference():
    preservation = trace("p", [[[0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0]]])
    edited = trace("e", [[[0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [1, 0], [1, 0], [1, 0]]])
    mask, _ = estimate_edit_token_mask(preservation, edited, [0], quantile=0.5, min_ratio=0.5, max_ratio=0.5)
    assert int(mask.sum()) == 2


def test_reward_selection_and_explicit_debug_candidate():
    source = Image.new("RGB", (2, 2))
    candidates = [Image.new("RGB", (2, 2), (i, i, i)) for i in range(4)]
    scorer = CallableRewardScorer(lambda **kwargs: kwargs["candidate"].getpixel((0, 0))[0])
    winner, rewards, used = select_winner(source, candidates, "edit", scorer)
    assert winner == 3 and rewards == [0.0, 1.0, 2.0, 3.0] and used
    winner, rewards, used = select_winner(source, candidates, "edit", None, candidate_index=1)
    assert winner == 1 and all(math.isnan(x) for x in rewards) and not used


def test_reward_missing_is_explicit_error():
    with pytest.raises(RewardUnavailable):
        select_winner(Image.new("RGB", (1, 1)), [Image.new("RGB", (1, 1)) for _ in range(4)], "edit", None)


def test_strength_endpoints_and_intermediate_continuity():
    preservation = trace("p", [[[0.0]], [[1.0]], [[2.0]]])
    winner = trace("e", [[[0.0]], [[2.0]], [[4.0]]])
    values = rollout_strengths(None, preservation, winner, [0.0, 0.5, 1.0])
    assert torch.equal(values[0.0], preservation.terminal)
    assert torch.equal(values[1.0], winner.terminal)
    assert torch.allclose(values[0.5], torch.tensor([[[-1.5]]]))


def test_strength_rejects_out_of_range():
    preservation = trace("p", [[[0.0]], [[1.0]]])
    winner = trace("e", [[[0.0]], [[2.0]]])
    with pytest.raises(ValueError):
        rollout_strengths(None, preservation, winner, [-0.1])
