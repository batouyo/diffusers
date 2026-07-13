from __future__ import annotations

import pytest
import torch

from interventions import TextBlockIntervention, assert_no_active_interventions, resolve_block
from probe_flux_kontext_blocks import packed_noise_latents


class ToyBlock(torch.nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, temb=None, image_rotary_emb=None, joint_attention_kwargs=None):
        del temb, image_rotary_emb, joint_attention_kwargs
        return encoder_hidden_states + 1, hidden_states + encoder_hidden_states.mean()


class ToyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([ToyBlock(), ToyBlock()])
        self.single_transformer_blocks = torch.nn.ModuleList([ToyBlock(), ToyBlock()])

    def forward(self, hidden_states, encoder_hidden_states):
        direct_inputs = []
        for block in list(self.transformer_blocks) + list(self.single_transformer_blocks):
            direct_inputs.append(encoder_hidden_states.clone())
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        return encoder_hidden_states, hidden_states, direct_inputs


@pytest.fixture
def values():
    return torch.ones(1, 3, 4), torch.ones(1, 2, 4)


def test_global_mapping():
    transformer = ToyTransformer()
    address, _ = resolve_block(transformer, 2)
    assert address.block_type == "single"
    assert address.local_index == 0
    with pytest.raises(IndexError):
        resolve_block(transformer, 4)


def test_alpha_one_is_exact_noop(values):
    encoder, hidden = values
    transformer = ToyTransformer()
    baseline = transformer(hidden.clone(), encoder.clone())[:2]
    with TextBlockIntervention(transformer, 1, "enhance_text", alpha=1.0) as intervention:
        observed = transformer(hidden.clone(), encoder.clone())[:2]
    assert intervention.call_count == 1
    assert all(torch.equal(left, right) for left, right in zip(baseline, observed))
    assert_no_active_interventions(transformer)


def test_only_target_input_is_directly_scaled(values):
    encoder, hidden = values
    transformer = ToyTransformer()
    with TextBlockIntervention(transformer, 1, "enhance_text", alpha=2.0):
        _, _, direct = transformer(hidden.clone(), encoder.clone())
    assert torch.equal(direct[0], encoder)
    # Block 0 adds one; block 1 receives that natural upstream state before its hook.
    assert torch.equal(direct[1], encoder + 1)
    # Downstream blocks naturally receive the causal result and are not expected to match baseline.
    assert_no_active_interventions(transformer)


def test_disable_zeros_only_target(values):
    encoder, hidden = values
    transformer = ToyTransformer()
    captured = {}

    def capture(module, args, kwargs):
        del module, args
        captured["encoder"] = kwargs["encoder_hidden_states"]

    handle = None
    try:
        with TextBlockIntervention(transformer, 0, "disable_text"):
            handle = transformer.transformer_blocks[0].register_forward_pre_hook(capture, with_kwargs=True)
            transformer(hidden.clone(), encoder.clone())
    finally:
        if handle is not None:
            handle.remove()
    assert torch.count_nonzero(captured["encoder"]) == 0


@pytest.mark.parametrize("global_index", [0, 2])
def test_remove_preserves_contract_and_restores(values, global_index):
    encoder, hidden = values
    transformer = ToyTransformer()
    _, block = resolve_block(transformer, global_index)
    assert "forward" not in block.__dict__
    with TextBlockIntervention(transformer, global_index, "remove_block") as intervention:
        out_encoder, out_hidden = block(hidden_states=hidden, encoder_hidden_states=encoder)
        assert torch.equal(out_encoder, encoder)
        assert torch.equal(out_hidden, hidden)
        assert out_encoder.dtype == encoder.dtype
        assert out_hidden.shape == hidden.shape
    assert intervention.call_count == 1
    assert "forward" not in block.__dict__
    assert_no_active_interventions(transformer)


def test_exception_restores_hook(values):
    encoder, hidden = values
    transformer = ToyTransformer()
    with pytest.raises(RuntimeError, match="boom"):
        with TextBlockIntervention(transformer, 0, "enhance_text", alpha=1.5):
            transformer(hidden.clone(), encoder.clone())
            raise RuntimeError("boom")
    assert_no_active_interventions(transformer)
    assert len(transformer.transformer_blocks[0]._forward_pre_hooks) == 0


def test_rejects_two_active_blocks(values):
    del values
    transformer = ToyTransformer()
    with TextBlockIntervention(transformer, 0, "enhance_text"):
        with pytest.raises(RuntimeError, match="another intervention"):
            with TextBlockIntervention(transformer, 1, "enhance_text"):
                pass


def test_multi_mode_is_explicit(values):
    encoder, hidden = values
    transformer = ToyTransformer()
    with TextBlockIntervention(transformer, 0, "enhance_text", allow_multi=True):
        with TextBlockIntervention(transformer, 2, "enhance_text", allow_multi=True):
            transformer(hidden.clone(), encoder.clone())
    assert_no_active_interventions(transformer)


def test_token_mask_broadcast(values):
    encoder, hidden = values
    transformer = ToyTransformer()
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    captured = {}

    def capture(module, args, kwargs):
        del module, args
        captured["encoder"] = kwargs["encoder_hidden_states"].clone()

    handle = None
    try:
        with TextBlockIntervention(transformer, 0, "enhance_text", alpha=2.0, token_mask=mask):
            handle = transformer.transformer_blocks[0].register_forward_pre_hook(capture, with_kwargs=True)
            transformer(hidden.clone(), encoder.clone())
    finally:
        if handle is not None:
            handle.remove()
    expected = encoder.clone()
    expected[:, [0, 2]] *= 2
    assert torch.equal(captured["encoder"], expected)


def test_same_seed_recreates_identical_initial_noise():
    class DummyPipe:
        vae_scale_factor = 8

        class transformer:
            class config:
                in_channels = 64

        @staticmethod
        def _pack_latents(latents, batch_size, channels, height, width):
            assert latents.shape == (batch_size, channels, height, width)
            return latents.clone()

    pipe = DummyPipe()
    first = packed_noise_latents(pipe, 64, 42, torch.float32, "cpu")
    repeated = packed_noise_latents(pipe, 64, 42, torch.float32, "cpu")
    different = packed_noise_latents(pipe, 64, 43, torch.float32, "cpu")
    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)
