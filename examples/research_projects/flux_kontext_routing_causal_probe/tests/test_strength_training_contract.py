import torch

from strength_overfit_data import assert_same_contract, input_contract
from strength_overfit_training import interpolated_teacher, teacher_difference_token_weights, velocity_losses


def test_contract_detects_changed_latent():
    common = dict(
        source_latents=torch.zeros(1, 2, 3),
        timestep=torch.tensor(500.0),
        sigma=torch.tensor(0.5),
        text_ids=torch.zeros(2, 3),
        image_ids=torch.zeros(4, 3),
        seed=1,
    )
    first = input_contract(target_latents=torch.zeros(1, 2, 3), **common)
    second = input_contract(target_latents=torch.ones(1, 2, 3), **common)
    try:
        assert_same_contract(first, second)
    except RuntimeError:
        pass
    else:
        raise AssertionError("contract mismatch was not detected")


def test_teacher_endpoints_and_monotonic_loss():
    neutral = torch.zeros(1, 2, 3)
    edit = torch.ones(1, 2, 3)
    assert torch.equal(interpolated_teacher(edit, neutral, 0.0), neutral)
    assert torch.equal(interpolated_teacher(edit, neutral, 1.0), edit)
    losses = velocity_losses(
        v_student=interpolated_teacher(edit, neutral, 0.25),
        v_edit=edit,
        v_neutral=neutral,
        strength=0.25,
        second_student=interpolated_teacher(edit, neutral, 0.75),
        second_strength=0.75,
        lambda_mono=1.0,
        lambda_progress=1.0,
    )
    assert losses["total"].item() < 1e-7


def test_teacher_difference_weights_are_detached_mean_one_and_optional():
    edit = torch.tensor([[[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]]], requires_grad=True)
    neutral = torch.zeros_like(edit, requires_grad=True)
    student = torch.full_like(edit, 0.25, requires_grad=True)
    weights = teacher_difference_token_weights(edit, neutral, background_weight=0.2)
    assert not weights.requires_grad
    torch.testing.assert_close(weights.mean(dim=1), torch.ones(1))
    plain = velocity_losses(v_student=student, v_edit=edit, v_neutral=neutral, strength=0.4)
    ones = velocity_losses(
        v_student=student,
        v_edit=edit,
        v_neutral=neutral,
        strength=0.4,
        token_weights=torch.ones_like(weights),
    )
    weighted = velocity_losses(
        v_student=student,
        v_edit=edit,
        v_neutral=neutral,
        strength=0.4,
        token_weights=weights,
    )
    torch.testing.assert_close(plain["velocity"], ones["velocity"])
    torch.testing.assert_close(plain["velocity_global"], ones["velocity_global"])
    assert weighted["velocity_global"].item() == plain["velocity_global"].item()
