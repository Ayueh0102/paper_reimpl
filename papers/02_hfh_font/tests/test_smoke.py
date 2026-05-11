"""HFH-Font smoke test.

Verifies the Phase-1 contract:
1. Build ``HFHFontModel`` from a tiny synthetic ``ModelConfig``.
2. Forward + backward + 1 optimizer step on a synthetic batch.
3. Loss is finite.
4. ``GaussianDiffusion.sample`` returns the expected latent shape.
"""

from __future__ import annotations

import torch
from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion
from paper_reimpl_shared.runner.smoke import make_synthetic_batch

from hfh_font.model import ModelConfig, build_model


def _make_cfg() -> ModelConfig:
    return ModelConfig(
        image_size=64,
        in_channels=1,
        content_channels=3,
        latent_channels=4,
        vae_down_factor=8,
        base_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(2,),
        d_ctx=64,
        n_heads=4,
        char_vocab_size=128,
        writer_vocab_size=16,
        script_vocab_size=4,
        n_refs=2,
        components_per_ref=4,
        dropout=0.0,
        diffusion_timesteps=100,
        diffusion_target="x0",
    )


def _make_batch(cfg: ModelConfig, device: torch.device) -> dict[str, torch.Tensor]:
    return make_synthetic_batch(
        batch_size=2,
        image_size=cfg.image_size,
        in_channels=cfg.in_channels,
        char_vocab_size=cfg.char_vocab_size,
        writer_vocab_size=cfg.writer_vocab_size,
        n_refs=cfg.n_refs,
        device=str(device),
    )


def _attach_ref_valid(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    batch = dict(batch)
    refs = batch["refs"]
    batch["ref_images"] = refs
    batch["ref_valid"] = torch.ones(refs.shape[0], refs.shape[1], dtype=torch.bool, device=refs.device)
    return batch


def test_smoke_build_and_train_step():
    torch.manual_seed(0)
    device = torch.device("cpu")
    cfg = _make_cfg()
    model = build_model(cfg).to(device)
    model.train()

    diffusion = GaussianDiffusion(
        timesteps=cfg.diffusion_timesteps,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=2e-2,
        prediction_target=cfg.diffusion_target,
        device=device,
    )

    batch = _attach_ref_valid(_make_batch(cfg, device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = model.compute_loss(batch, diffusion, cfg_dropout=0.0)
    loss = losses["loss"]
    assert torch.isfinite(loss), f"non-finite loss: {loss}"

    optimizer.zero_grad()
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "no parameter received gradient"

    optimizer.step()
    # One more forward to make sure the step didn't NaN anything out.
    losses2 = model.compute_loss(batch, diffusion, cfg_dropout=0.0)
    assert torch.isfinite(losses2["loss"]), "loss became non-finite after optimizer step"


def test_smoke_sample_shape():
    torch.manual_seed(0)
    device = torch.device("cpu")
    cfg = _make_cfg()
    model = build_model(cfg).to(device).eval()
    diffusion = GaussianDiffusion(
        timesteps=cfg.diffusion_timesteps,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=2e-2,
        prediction_target=cfg.diffusion_target,
        device=device,
    )
    batch = _attach_ref_valid(_make_batch(cfg, device))
    # Shrink diffusion to 5 steps to keep test fast.
    short_diffusion = GaussianDiffusion(
        timesteps=5,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=2e-2,
        prediction_target=cfg.diffusion_target,
        device=device,
    )
    latent_h = cfg.image_size // cfg.vae_down_factor
    shape = (batch["image"].shape[0], cfg.latent_channels, latent_h, latent_h)
    out = short_diffusion.sample(
        model,
        shape=shape,
        content=batch["content"],
        char_id=batch["char_id"],
        script_id=batch["script_id"],
        writer_id=batch["writer_id"],
        ref_images=batch["ref_images"],
        ref_valid=batch["ref_valid"],
        sampler="ddim",
        cfg_scale=1.0,
        cfg_uncond_drops_content=False,
        device=device,
    )
    del diffusion  # unused beyond shape check
    assert out.shape == shape, f"sample shape mismatch: {out.shape} != {shape}"
    assert torch.isfinite(out).all(), "sample contains non-finite values"


def test_smoke_sds_path():
    """Verify SDS-loss path runs end-to-end and produces gradient.

    Zero-init AdaLN-Zero + zero-init out-conv means student and teacher both
    emit exactly 0 at construction time, so the SDS loss is 0 at the very
    first call. We do one regular ``compute_loss`` step to break the
    student/teacher symmetry, then verify the SDS gradient is finite and
    non-zero.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")
    cfg = _make_cfg()
    student = build_model(cfg).to(device)
    teacher = build_model(cfg).to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.train()
    diffusion = GaussianDiffusion(
        timesteps=cfg.diffusion_timesteps,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=2e-2,
        prediction_target=cfg.diffusion_target,
        device=device,
    )
    batch = _attach_ref_valid(_make_batch(cfg, device))

    # Warm-up: regular denoising step to break zero-init symmetry between
    # student and teacher. Without this, both emit 0 and SDS loss is exactly
    # 0 with zero gradient — that's a real property of zero-init nets, not a
    # bug in the SDS plumbing.
    warmup_opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    warmup_losses = student.compute_loss(batch, diffusion, cfg_dropout=0.0)
    warmup_opt.zero_grad()
    warmup_losses["loss"].backward()
    warmup_opt.step()

    student.zero_grad(set_to_none=True)
    losses = student.compute_sds_loss(batch, teacher, diffusion)
    assert torch.isfinite(losses["loss"]), "SDS loss non-finite"
    losses["loss"].backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
    assert has_grad, "student got no SDS gradient after warmup"
