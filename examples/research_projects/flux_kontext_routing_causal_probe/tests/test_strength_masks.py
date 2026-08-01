import torch

from strength_overfit_masks import TemporalMaskEMA, make_mask, robust_normalize_mask, token_mask_to_image


def test_degenerate_mask_falls_back_to_one():
    normalized, degenerate = robust_normalize_mask(torch.ones(2, 6))
    assert degenerate.tolist() == [True, True]
    assert torch.equal(normalized, torch.ones_like(normalized))


def test_hard_soft_and_ema():
    edit = torch.tensor([[[2.0, 0.0], [0.0, 0.0]]])
    neutral = torch.zeros_like(edit)
    hard = make_mask(edit, neutral, mask_type="hard", tau=0.5, temperature=0.1, lambda_bg=0.2)
    soft = make_mask(edit, neutral, mask_type="soft", tau=0.5, temperature=0.1, lambda_bg=0.2)
    assert hard.weight.shape == (1, 2, 1)
    assert hard.weight[0, 0, 0] >= hard.weight[0, 1, 0]
    assert soft.weight[0, 0, 0] > soft.weight[0, 1, 0]
    ema = TemporalMaskEMA(beta=0.7)
    first = ema.update(torch.ones(1, 2))
    second = ema.update(torch.zeros(1, 2))
    assert torch.equal(first, torch.ones(1, 2))
    assert torch.allclose(second, torch.full((1, 2), 0.7))


def test_token_mask_to_image():
    image = token_mask_to_image(torch.arange(4.0).view(1, 4), 16, 16, 8)
    assert image.shape == (1, 1, 16, 16)

