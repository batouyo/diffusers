import torch
from torch import nn

from strength_residual import StrengthResidualIntervention, TargetStrengthResidual, TimeStrengthGate


class Block(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, **_kwargs):
        return encoder_hidden_states + 1, hidden_states + 1


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([Block(), Block()])
        self.single_transformer_blocks = nn.ModuleList([])


def test_adapter_zero_initialization_and_gate_initial_value():
    adapter = TargetStrengthResidual(hidden_size=8, rank=2)
    hidden = torch.randn(2, 3, 8)
    assert torch.equal(adapter(hidden), torch.zeros_like(hidden))
    gate = TimeStrengthGate(num_layers=3)
    assert torch.equal(gate(0.5, 0.25, 2, hidden.device), torch.ones(2, 3))


def test_strength_one_is_bit_exact_and_target_only():
    transformer = Transformer()
    intervention = StrengthResidualIntervention(
        transformer, ["dual.00", "dual.01"], target_tokens=3, hidden_size=8, rank=2
    )
    image = torch.randn(1, 6, 8)
    text = torch.randn(1, 2, 8)
    baseline = image
    for block in transformer.transformer_blocks:
        _, baseline = block(baseline, text)

    intervention.set_context(strength=1.0, sigma=0.5)
    intervention.reset_sequence()
    with intervention.applied():
        controlled = image
        for block in transformer.transformer_blocks:
            _, controlled = block(controlled, text)
    assert torch.equal(controlled, baseline)

    with torch.no_grad():
        intervention.adapter("dual.00").up.fill_(0.25)
        intervention.adapter("dual.01").up.fill_(0.25)
    intervention.set_context(strength=0.0, sigma=0.5)
    intervention.reset_sequence()
    with intervention.applied():
        controlled = image
        for block in transformer.transformer_blocks:
            _, controlled = block(controlled, text)
    assert torch.equal(controlled[:, 3:], baseline[:, 3:])
    assert not torch.equal(controlled[:, :3], baseline[:, :3])


def test_invalid_spatial_weight_is_rejected():
    transformer = Transformer()
    intervention = StrengthResidualIntervention(transformer, ["dual.00"], target_tokens=3, hidden_size=8, rank=2)
    intervention.set_context(strength=0.0, sigma=0.5, spatial_weight=torch.ones(1, 2))
    with intervention.applied():
        try:
            transformer.transformer_blocks[0](torch.randn(1, 6, 8), torch.randn(1, 2, 8))
        except ValueError as error:
            assert "spatial_weight" in str(error)
        else:
            raise AssertionError("expected a spatial weight validation error")

