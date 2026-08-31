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
    regional_velocity,
    strength_step,
    velo_edit_velocity,
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
    x = torch.zeros(1, 2, 1)
    vp = torch.ones_like(x)
    ve = torch.full_like(x, 3.0)
    s0 = strength_step(x, vp, ve, 1.0, 0.5, 0.0)
    s1 = strength_step(x, vp, ve, 1.0, 0.5, 1.0)
    s99 = strength_step(x, vp, ve, 1.0, 0.5, 0.99)
    assert torch.allclose(s0, torch.full_like(x, -0.5))
    assert torch.allclose(s1, torch.full_like(x, -1.5))
    assert torch.max(torch.abs(s1 - s99)) < 0.1


def test_velocity_windows_are_independent():
    z_t = torch.zeros(1, 4, 1)
    z_source = torch.full_like(z_t, -1.0)
    v_edit = torch.tensor([[[1.0], [3.0], [1.0], [3.0]]])
    output, edit, preserve, v_ref = velo_edit_velocity(
        z_t,
        z_source,
        v_edit,
        1.0,
        0.5,
        threshold=0.8,
        preserve_active=True,
        edit_strength_active=False,
    )
    assert torch.equal(output[preserve], v_ref[preserve].to(output.dtype))
    assert torch.equal(output[edit], v_edit[edit])

    blended, _, _ = regional_velocity(
        v_edit,
        v_ref,
        torch.tensor([[0.25]]),
        torch.zeros_like(v_edit),
        preserve_active=False,
        edit_strength_active=True,
    )
    assert torch.allclose(blended, 0.75 * v_ref + 0.25 * v_edit)


def test_config_uses_short_strength_window_and_cpu_generator():
    cfg = ContinuousStrengthConfig()
    assert cfg.preserve_step_count == 4
    assert cfg.edit_strength_step_count == 2
    assert cfg.critical_steps == 1
    assert cfg.generator_device == "cpu"
    with pytest.raises(ValueError):
        ContinuousStrengthConfig(edit_strength_step_count=5)


def test_strength_rejects_out_of_range():
    preservation = trace("p", [[[0.0]], [[1.0]]])
    winner = trace("e", [[[0.0]], [[2.0]]])
    with pytest.raises(ValueError):
        rollout_strengths(None, preservation, winner, [-0.1])
