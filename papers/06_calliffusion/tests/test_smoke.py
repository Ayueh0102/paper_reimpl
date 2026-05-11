"""Smoke tests for calliffusion blind reimplementation.

Verifies — using offline stubs only (no transformers download required):
1. The U-Net + text encoder builds.
2. Forward + backward + 1 optimizer step on synthetic data produces finite loss.
3. The LoRA wrapper actually zeros out at init and trains thereafter.
4. The shared diffusion utility integrates with our forward signature.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion

from calliffusion.dataset import (
    SyntheticPromptDataset,
    collate_prompt_batch,
)
from calliffusion.lora import apply_lora_to_module, freeze_non_lora, lora_parameters
from calliffusion.model import CalliffusionUNet, CalliffusionUNetConfig
from calliffusion.sample import sample_prompts
from calliffusion.text import StubTextEncoder


def _build_tiny_model() -> tuple[CalliffusionUNet, StubTextEncoder]:
    cfg = CalliffusionUNetConfig(
        image_size=16,
        in_channels=1,
        out_channels=1,
        base_channels=16,
        channel_mult=[1, 2],
        num_res_blocks=1,
        time_emb_dim=32,
        context_dim=32,
        num_heads=2,
        dropout=0.0,
    )
    unet = CalliffusionUNet(cfg)
    text = StubTextEncoder(hidden_size=32, max_length=8)
    return unet, text


def test_unet_forward_shape() -> None:
    """U-Net output shape matches input shape."""
    unet, text = _build_tiny_model()
    out = text.encode(["a b c", "d e f"])
    x = torch.randn(2, 1, 16, 16)
    t = torch.randint(0, 1000, (2,))
    pred = unet(x, t, context=out.last_hidden_state, context_mask=out.attention_mask)
    assert pred.shape == x.shape
    assert torch.isfinite(pred).all()


def test_smoke() -> None:
    """Full forward + backward + 1 optimizer step, all finite."""
    torch.manual_seed(0)
    unet, text = _build_tiny_model()
    text.add_special_tokens(["writer0", "writer1"])  # exercise vocab growth
    diffusion = GaussianDiffusion(
        timesteps=100,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
        prediction_target="epsilon",
        device="cpu",
    )

    ds = SyntheticPromptDataset(length=4, image_size=16, writer_vocab_size=2, char_vocab_size=4)
    batch = collate_prompt_batch([ds[i] for i in range(4)])

    ctx_out = text.encode(batch["prompt"])
    d_batch = diffusion.sample_training_batch(batch["image"])
    pred = unet(
        d_batch.x_t,
        d_batch.timesteps,
        context=ctx_out.last_hidden_state,
        context_mask=ctx_out.attention_mask,
    )
    loss = F.mse_loss(pred, d_batch.target)
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    assert math.isfinite(float(loss.item()))

    params = [p for p in unet.parameters() if p.requires_grad] + list(text.parameters())
    optim = torch.optim.AdamW(params, lr=1e-4)
    optim.zero_grad()
    loss.backward()
    # at least one gradient should be non-zero
    any_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in params if p.requires_grad)
    assert any_grad, "no gradient flowed back to any parameter"
    optim.step()


def test_lora_zero_at_init_and_trains() -> None:
    """LoRA must be a no-op at init, and the LoRA params must accumulate gradients."""
    unet, text = _build_tiny_model()
    out = text.encode(["a b", "c d"])
    x = torch.randn(2, 1, 16, 16)
    t = torch.tensor([5, 50], dtype=torch.long)
    # Un-zero conv_out so the gradient signal reaches LoRA params (zero-init
    # conv_out is intentional for training stability — paper_notes §2 — but
    # would block the gradient test here).
    with torch.no_grad():
        unet.conv_out.weight.normal_(std=0.05)
        unet.conv_out.bias.normal_(std=0.05)
    before = unet(x, t, context=out.last_hidden_state, context_mask=out.attention_mask).clone()
    n_wrapped = apply_lora_to_module(unet, rank=2, alpha=4.0)
    assert n_wrapped > 0, "expected at least one LoRA wrap"
    after = unet(x, t, context=out.last_hidden_state, context_mask=out.attention_mask)
    assert torch.allclose(before, after, atol=1e-5), "LoRA must be no-op at init"

    freeze_non_lora(unet)
    lora_ps = lora_parameters(unet)
    assert len(lora_ps) == 2 * n_wrapped, "every LoraLinear should expose A and B"
    # Trainable param count after freeze should equal len(lora_ps)
    trainables = [p for p in unet.parameters() if p.requires_grad]
    assert len(trainables) == len(lora_ps)

    # Backward at step 0: A has zero gradient (because B=0), but B has non-zero
    # gradient (because A is kaiming-initialised). One SGD step moves B off zero
    # so the adapter is no longer the identity, and the model output changes.
    pred = unet(x, t, context=out.last_hidden_state, context_mask=out.attention_mask)
    target = pred.detach() + 1.0
    loss = F.mse_loss(pred, target)
    loss.backward()
    # B is shape [out, rank]; identify it as the params with rank as last dim == 2.
    b_params = [p for p in lora_ps if p.ndim == 2 and p.shape[1] == 2]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in b_params), (
        "LoRA B matrices received no gradient at step 0"
    )
    optim = torch.optim.SGD(lora_ps, lr=1e-1)
    optim.step()
    after2 = unet(x, t, context=out.last_hidden_state, context_mask=out.attention_mask)
    assert not torch.allclose(after, after2, atol=1e-6), "LoRA params did not update model output"


def test_sample_runs_one_prompt() -> None:
    """End-to-end sampling on a tiny model finishes and returns finite tensor."""
    unet, text = _build_tiny_model()
    diffusion = GaussianDiffusion(timesteps=4, prediction_target="epsilon", device="cpu")
    out = sample_prompts(
        unet,
        text,
        diffusion,
        prompts=["a b c"],
        shape=(1, 1, 16, 16),
        sampler="ddim",
        cfg_scale=1.0,
        device="cpu",
    )
    assert out.shape == (1, 1, 16, 16)
    assert torch.isfinite(out).all()
    assert out.min() >= -1.0 - 1e-5 and out.max() <= 1.0 + 1e-5


@pytest.mark.parametrize("cfg_scale", [1.0, 3.0])
def test_sample_cfg(cfg_scale: float) -> None:
    """CFG branch must run for both scale=1 and scale>1."""
    unet, text = _build_tiny_model()
    diffusion = GaussianDiffusion(timesteps=3, prediction_target="epsilon", device="cpu")
    out = sample_prompts(
        unet,
        text,
        diffusion,
        prompts=["hello"],
        shape=(1, 1, 16, 16),
        sampler="ddpm",
        cfg_scale=cfg_scale,
        device="cpu",
    )
    assert torch.isfinite(out).all()
