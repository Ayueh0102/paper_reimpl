"""Smoke test for DP-Font blind reimplementation.

Verifies:
  1. Model forward + backward + 1 optimizer step produces finite loss on a
     tiny synthetic batch (CPU-only, no disk I/O).
  2. PINN loss is differentiable and contributes a non-zero gradient back
     into the U-Net parameters (the DL-review rubric flags this as the
     critical correctness check for DP-Font).
  3. The shared sampler accepts the DP-Font model with the frozen-condition
     adapter from ``sample.py``.
"""

from __future__ import annotations

import torch

from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion
from paper_reimpl_shared.runner.smoke import make_synthetic_batch

from dp_font.dataset import collate_dp_font_batch, synthesise_stroke_order
from dp_font.model import DPFontConfig, build_dp_font
from dp_font.pinn_losses import pinn_loss
from dp_font.sample import sample as dp_font_sample
from dp_font.train import compute_loss


def _tiny_config(image_size: int, content_channels: int) -> DPFontConfig:
    """Tiny model so the test stays under a few seconds on CPU."""
    return DPFontConfig(
        image_size=image_size,
        in_channels=1,
        content_channels=content_channels,
        base_channels=16,
        channel_mult=(1, 2, 2),
        attn_resolutions=(8,),
        num_res_blocks=1,
        time_embed_dim=64,
        cond_embed_dim=64,
        num_heads=2,
        dropout=0.0,
        writer_vocab_size=8,
        script_vocab_size=4,
        char_vocab_size=64,
        stroke_vocab_size=12,
        stroke_seq_len=8,
        use_ink_intensity=True,
        use_font_size=True,
    )


def _augment_batch(batch: dict, cfg: DPFontConfig) -> dict:
    """Add DP-Font's extra fields to the shared synthetic batch."""
    bs = batch["image"].shape[0]
    stroke = torch.tensor(
        [
            synthesise_stroke_order(
                seed_text=f"smoke::{i}", vocab_size=cfg.stroke_vocab_size, seq_len=cfg.stroke_seq_len
            )
            for i in range(bs)
        ],
        dtype=torch.long,
    )
    batch["stroke_order"] = stroke
    batch["ink_intensity"] = torch.linspace(0.1, 0.9, bs)
    batch["font_size"] = torch.linspace(0.2, 0.8, bs)
    return batch


def test_smoke_forward_backward_step() -> None:
    """End-to-end smoke: model + L_simple + L_PINN trains 1 step on CPU."""
    torch.manual_seed(0)
    image_size = 32
    batch = make_synthetic_batch(
        batch_size=2,
        image_size=image_size,
        in_channels=1,
        n_refs=0,
        device="cpu",
    )
    cfg = _tiny_config(image_size=image_size, content_channels=batch["content"].shape[1])
    batch = _augment_batch(batch, cfg)

    model = build_dp_font(cfg)
    diffusion = GaussianDiffusion(
        timesteps=50,
        beta_schedule="cosine",
        prediction_target="epsilon",
        device="cpu",
    )
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    loss, log = compute_loss(
        model=model,
        diffusion=diffusion,
        batch=batch,
        pinn_weight=0.1,
        pinn_weights={"weight_diffusion": 1.0, "weight_nib": 1.0, "weight_continuity": 1.0, "nu": 1.0},
        cfg_drop_prob=0.1,
        skeleton_channel_index=None,
    )

    assert torch.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"
    assert log["loss_pinn"] >= 0.0
    loss.backward()

    # Every branch — content encoder, guidance head, U-Net trunk — must
    # receive non-zero gradient. Catches "conditioning path disconnected".
    grad_norms = {
        "content_encoder": sum(
            p.grad.norm().item()
            for p in model.content_encoder.parameters()
            if p.grad is not None
        ),
        "guidance": sum(
            p.grad.norm().item()
            for p in model.guidance.parameters()
            if p.grad is not None
        ),
        "unet": sum(
            p.grad.norm().item() for p in model.unet.parameters() if p.grad is not None
        ),
    }
    for branch, norm in grad_norms.items():
        assert norm > 0.0, f"branch={branch} got zero gradient — conditioning may be broken"

    optim.step()
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all().item(), f"Non-finite param after step: {name}"
    assert log["loss_total"] >= 0.0


def test_pinn_loss_is_differentiable() -> None:
    """Each PINN sub-term must produce a gradient on the predicted x0.

    This is the DP-Font-specific must-have (DL rubric: 'PINN 物理 loss
    真有反向傳回 generator'). We verify by constructing a fake x0_pred that
    requires grad, running each sub-term, and asserting the autograd graph
    is non-trivial.
    """
    torch.manual_seed(1)
    x0 = torch.randn(2, 1, 16, 16, requires_grad=True)
    total, log = pinn_loss(
        x0,
        skeleton=None,
        weight_diffusion=1.0,
        weight_nib=1.0,
        weight_continuity=1.0,
        nu=1.0,
    )
    assert torch.isfinite(total).item()
    assert log["loss_pinn_diffusion"] >= 0.0
    assert log["loss_pinn_nib"] >= 0.0
    assert log["loss_pinn_continuity"] >= 0.0
    total.backward()
    assert x0.grad is not None, "x0 received no PINN gradient"
    assert torch.isfinite(x0.grad).all().item()
    assert x0.grad.abs().sum().item() > 0.0, "PINN gradient is exactly zero"


def test_pinn_contributes_to_unet_gradient() -> None:
    """When ``pinn_weight > 0`` the U-Net must receive gradient from L_PINN
    (not only from L_simple). Compare grad norms with and without PINN.
    """
    torch.manual_seed(2)
    image_size = 32
    base_batch = make_synthetic_batch(batch_size=2, image_size=image_size, in_channels=1, n_refs=0)
    cfg = _tiny_config(image_size=image_size, content_channels=base_batch["content"].shape[1])

    # ---- Run 1: pinn_weight=0 (control) ----
    model_a = build_dp_font(cfg)
    diffusion = GaussianDiffusion(
        timesteps=50, beta_schedule="cosine", prediction_target="epsilon", device="cpu"
    )
    torch.manual_seed(42)
    batch_a = _augment_batch(dict(base_batch), cfg)
    loss_a, _ = compute_loss(
        model=model_a, diffusion=diffusion, batch=batch_a, pinn_weight=0.0,
    )
    loss_a.backward()
    grad_a = sum(p.grad.norm().item() for p in model_a.unet.parameters() if p.grad is not None)

    # ---- Run 2: pinn_weight=10.0 (PINN dominant) ----
    model_b = build_dp_font(cfg)
    # Copy weights from model_a so the only difference is the loss term.
    model_b.load_state_dict(model_a.state_dict())
    torch.manual_seed(42)
    batch_b = _augment_batch(dict(base_batch), cfg)
    loss_b, log_b = compute_loss(
        model=model_b, diffusion=diffusion, batch=batch_b, pinn_weight=10.0,
    )
    loss_b.backward()
    grad_b = sum(p.grad.norm().item() for p in model_b.unet.parameters() if p.grad is not None)

    assert grad_a > 0.0
    assert grad_b > 0.0
    # With λ_PINN=10 vs 0 there should be a measurable change in the
    # U-Net gradient norm — proves PINN actually back-props.
    assert abs(grad_b - grad_a) > 1e-6, (
        f"PINN did not influence U-Net gradient (grad_a={grad_a:.6f} grad_b={grad_b:.6f})"
    )
    assert log_b["loss_pinn"] > 0.0


def test_sampler_runs() -> None:
    """The shared sampler must accept the DP-Font model + frozen-cond adapter."""
    torch.manual_seed(3)
    image_size = 32
    batch = make_synthetic_batch(batch_size=2, image_size=image_size, in_channels=1, n_refs=0)
    cfg = _tiny_config(image_size=image_size, content_channels=batch["content"].shape[1])
    batch = _augment_batch(batch, cfg)
    model = build_dp_font(cfg).eval()
    diffusion = GaussianDiffusion(
        timesteps=4, beta_schedule="cosine", prediction_target="epsilon", device="cpu"
    )
    out = dp_font_sample(
        model=model,
        diffusion=diffusion,
        content=batch["content"],
        writer_id=batch["writer_id"],
        script_id=batch["script_id"],
        char_id=batch["char_id"],
        stroke_order=batch["stroke_order"],
        ink_intensity=batch["ink_intensity"],
        font_size=batch["font_size"],
        sampler="ddpm",
        cfg_scale=1.5,
        device="cpu",
    )
    assert out.shape == (2, 1, image_size, image_size)
    assert torch.isfinite(out).all().item()


def test_collate_round_trips_stroke_order() -> None:
    """Verify the picklable collate emits stroke_order as a long tensor."""
    cfg = _tiny_config(image_size=16, content_channels=1)
    items = []
    for i in range(3):
        items.append(
            {
                "image": torch.zeros(1, 16, 16),
                "content": torch.zeros(1, 16, 16),
                "char_id": i,
                "script_id": 0,
                "writer_id": 0,
                "style_family_id": 0,
                "unit_id": 0,
                "ref_images": [],
                "metadata": {"i": i},
                "stroke_order": synthesise_stroke_order(
                    seed_text=f"item::{i}",
                    vocab_size=cfg.stroke_vocab_size,
                    seq_len=cfg.stroke_seq_len,
                ),
                "ink_intensity": 0.5,
                "font_size": 0.5,
            }
        )
    out = collate_dp_font_batch(items, max_refs=0)
    assert out["stroke_order"].shape == (3, cfg.stroke_seq_len)
    assert out["stroke_order"].dtype == torch.long
    assert out["ink_intensity"].shape == (3,)
    assert out["font_size"].shape == (3,)
