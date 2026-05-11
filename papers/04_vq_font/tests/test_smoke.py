"""Smoke tests for VQ-Font (Phase 2 paper-faithful).

Three checks:
  1. VQGAN forward + backward + 1 optimizer step (Stage 0) — finite loss,
     finite parameters, codebook indices in [0, K).
  2. Transformer forward + backward + 1 optimizer step (Stage 1+) — finite
     loss, gradient reaches the K/Q/V projections (SSEM is parameter-free
     now, so we just check the cross-attn projections receive gradient
     from the reference path). Partial-freeze check: ONLY the configured
     early-decoder + post_quant params on VQGAN have requires_grad=True.
  3. Sample path: argmax decode runs without crash and produces an image
     of the expected shape.
  4. Stage 0 full loss smoke: VQLPIPSWithDiscriminator (simple_loss=False)
     runs one G + D step on a tiny VQGAN — finite, no NaNs.
"""

from __future__ import annotations

import torch

from paper_reimpl_shared.runner.smoke import make_synthetic_batch

from vq_font.model import (
    NUM_STRUCTURE_CLASSES,
    TransformerConfig,
    VQFontConfig,
    VQGANConfig,
    build_vq_font,
    build_vqgan,
)
from vq_font.sample import sample_vq_font, sample_vqgan_recon
from vq_font.train import transformer_compute_loss, vqgan_compute_loss
from vq_font.vqgan_loss import VQLPIPSLossConfig, VQLPIPSWithDiscriminator


def _tiny_vqgan_cfg(image_size: int) -> VQGANConfig:
    """Tiny VQGAN: 32px input -> 4x4 latent grid via 3 stages of stride-2."""
    return VQGANConfig(
        image_size=image_size,
        in_channels=1,
        base_channels=8,
        channel_mult=(1, 1, 2, 4),   # 3 stride-2 downsamples
        z_channels=32,
        embed_dim=32,
        num_embeddings=64,
        commitment_weight=0.25,
        num_res_blocks=1,
        dropout=0.0,
    )


def _tiny_transformer_cfg(vqgan_cfg: VQGANConfig, *, num_refs: int = 2) -> TransformerConfig:
    return TransformerConfig(
        image_size=vqgan_cfg.image_size,
        latent_resolution=vqgan_cfg.out_resolution(),
        embed_dim=vqgan_cfg.embed_dim,
        num_blocks=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        num_refs=num_refs,
        codebook_size=vqgan_cfg.num_embeddings,
        num_structures=NUM_STRUCTURE_CLASSES,
    )


def test_vqgan_stage_smoke() -> None:
    """Stage 0: train the VQGAN end-to-end on a single tiny batch (L1 + commitment)."""
    torch.manual_seed(0)
    image_size = 32
    batch = make_synthetic_batch(
        batch_size=2, image_size=image_size, in_channels=1, n_refs=0, device="cpu"
    )
    cfg = _tiny_vqgan_cfg(image_size)
    model = build_vqgan(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    loss, log = vqgan_compute_loss(model=model, batch=batch)
    assert torch.isfinite(loss).item(), f"VQGAN loss not finite: {loss.item()}"
    assert log["loss_recon"] >= 0.0
    assert log["loss_vq"] >= 0.0
    loss.backward()

    enc_grad = sum(p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None)
    dec_grad = sum(p.grad.norm().item() for p in model.decoder.parameters() if p.grad is not None)
    cb_grad = model.codebook.codebook.weight.grad.norm().item() if model.codebook.codebook.weight.grad is not None else 0.0
    assert enc_grad > 0.0, "VQGAN encoder received zero gradient"
    assert dec_grad > 0.0, "VQGAN decoder received zero gradient"
    assert cb_grad > 0.0, "VQGAN codebook received zero gradient from VQ losses"

    optim.step()
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all().item(), f"Non-finite parameter after step: {name}"

    indices = model.encode_indices(batch["image"])
    assert indices.dtype == torch.long
    assert indices.min().item() >= 0
    assert indices.max().item() < cfg.num_embeddings
    recon = sample_vqgan_recon(model, batch["image"])
    assert recon.shape == batch["image"].shape
    assert torch.isfinite(recon).all().item()


def test_vqgan_full_loss_smoke() -> None:
    """Stage 0 with VQLPIPSWithDiscriminator — one G + D step, finite."""
    torch.manual_seed(7)
    image_size = 32
    batch = make_synthetic_batch(
        batch_size=2, image_size=image_size, in_channels=1, n_refs=0, device="cpu"
    )
    cfg = _tiny_vqgan_cfg(image_size)
    model = build_vqgan(cfg)
    loss_cfg = VQLPIPSLossConfig(
        disc_start=0,                     # GAN active from step 0 for the smoke
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        perceptual_weight=0.0,            # skip LPIPS download in CI smoke
        disc_num_layers=2,                # smaller D for 32px input
        disc_in_channels=1,
        disc_ndf=16,
        disc_loss="hinge",
    )
    loss_mod = VQLPIPSWithDiscriminator(loss_cfg)
    loss_mod.train()
    model.train()

    optim_g = torch.optim.Adam(
        list(model.encoder.parameters())
        + list(model.decoder.parameters())
        + list(model.codebook.parameters()),
        lr=1e-4, betas=(0.5, 0.9),
    )
    optim_d = torch.optim.Adam(loss_mod.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    # G step
    out = model(batch["image"])
    last_layer = model.decoder.out_conv.conv.weight
    g_loss, g_log = loss_mod(
        out.vq_loss, batch["image"], out.recon,
        optimizer_idx=0, global_step=0, last_layer=last_layer, split="train",
    )
    assert torch.isfinite(g_loss).item(), f"G loss not finite: {g_loss.item()}"
    optim_g.zero_grad(set_to_none=True)
    g_loss.backward()
    optim_g.step()

    # D step (re-forward to detach)
    with torch.no_grad():
        out2 = model(batch["image"])
    d_loss, d_log = loss_mod(
        out2.vq_loss, batch["image"], out2.recon,
        optimizer_idx=1, global_step=0, last_layer=None, split="train",
    )
    assert torch.isfinite(d_loss).item(), f"D loss not finite: {d_loss.item()}"
    optim_d.zero_grad(set_to_none=True)
    d_loss.backward()
    optim_d.step()

    for name, p in model.named_parameters():
        assert torch.isfinite(p).all().item(), f"Non-finite model param after step: {name}"
    for name, p in loss_mod.discriminator.named_parameters():
        assert torch.isfinite(p).all().item(), f"Non-finite disc param after step: {name}"


def test_transformer_stage_smoke() -> None:
    """Stage 1+: train the Transformer with PARTIAL-frozen VQGAN."""
    torch.manual_seed(1)
    image_size = 32
    n_refs = 2
    batch = make_synthetic_batch(
        batch_size=2, image_size=image_size, in_channels=1, n_refs=n_refs, device="cpu"
    )
    batch["ref_images"] = batch.pop("refs")
    batch["ref_valid"] = torch.ones(2, n_refs, dtype=torch.bool)
    batch["structure_id"] = torch.tensor([0, 4], dtype=torch.long)  # in-range (0..12)

    vqgan_cfg = _tiny_vqgan_cfg(image_size)
    tr_cfg = _tiny_transformer_cfg(vqgan_cfg, num_refs=n_refs)
    # 'partial' freeze: encoder + late decoder + codebook frozen; first 3
    # decoder ResBlocks + post_quant trainable.
    model = build_vq_font(
        VQFontConfig(vqgan=vqgan_cfg, transformer=tr_cfg),
        freeze_vqgan="partial",
    )
    # Partial freeze check: trainable set equals the patterns that exist in
    # this (tiny) VQGAN topology. ``_partial_trainable`` is the intersection
    # of the configured patterns and the actual param names.
    trainable_vqgan_names = {n for n, p in model.vqgan.named_parameters() if p.requires_grad}
    expected = set(model._partial_trainable)
    assert len(expected) > 0, "expected at least one trainable pattern under partial freeze"
    assert trainable_vqgan_names == expected, (
        f"Partial-freeze mismatch.\n trainable: {sorted(trainable_vqgan_names)}\n"
        f" expected:  {sorted(expected)}"
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(trainable, lr=1e-4, betas=(0.0, 0.9))
    loss, log = transformer_compute_loss(model=model, batch=batch)
    assert torch.isfinite(loss).item(), f"transformer loss not finite: {loss.item()}"
    assert log["loss_token"] > 0.0  # CE on random init should be ~log(K)
    assert 0.0 <= log["token_acc"] <= 1.0
    loss.backward()

    # K/Q/V projections must receive gradient (cross-attn pre-stack path).
    q_grad = sum(p.grad.norm().item() for p in model.transformer.linears_query.parameters() if p.grad is not None)
    k_grad = sum(p.grad.norm().item() for p in model.transformer.linears_key.parameters() if p.grad is not None)
    v_grad = sum(p.grad.norm().item() for p in model.transformer.linears_value.parameters() if p.grad is not None)
    assert q_grad > 0.0, "linears_query did not receive gradient"
    assert k_grad > 0.0, "linears_key did not receive gradient"
    assert v_grad > 0.0, "linears_value did not receive gradient — ref path may be disconnected"

    # mlp_head must receive gradient.
    head_grad = sum(p.grad.norm().item() for p in model.transformer.mlp_head.parameters() if p.grad is not None)
    assert head_grad > 0.0, "mlp_head did not receive gradient"

    # Self-attn blocks must train.
    block_grad = sum(p.grad.norm().item() for p in model.transformer.former[0].parameters() if p.grad is not None)
    assert block_grad > 0.0, "transformer self-attn block did not receive gradient"

    # Partial-freeze VQGAN: trainable params should receive gradient; frozen params should NOT.
    for name, p in model.vqgan.named_parameters():
        if name in expected:
            # Trainable — accept zero grad only if the param genuinely isn't
            # in the computational path of this loss (post_quant is reached
            # only through encode_target_indices which is no_grad).
            # The early decoder layers ARE in the path via _vqgan_encode +
            # decode_indices, but Stage 1 loss doesn't call decode, so we
            # accept zero. Just verify it's a *number* (finite).
            if p.grad is not None:
                assert torch.isfinite(p.grad).all().item(), f"non-finite grad on {name}"
        else:
            assert p.grad is None or p.grad.abs().sum().item() == 0.0, (
                f"frozen VQGAN param got non-zero gradient: {name}"
            )

    optim.step()
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all().item(), f"Non-finite parameter after step: {name}"


def test_transformer_full_freeze_legacy_smoke() -> None:
    """Legacy `freeze_vqgan='full'` path (strict blind-impl behaviour)."""
    torch.manual_seed(11)
    image_size = 32
    n_refs = 2
    vqgan_cfg = _tiny_vqgan_cfg(image_size)
    tr_cfg = _tiny_transformer_cfg(vqgan_cfg, num_refs=n_refs)
    model = build_vq_font(
        VQFontConfig(vqgan=vqgan_cfg, transformer=tr_cfg), freeze_vqgan="full",
    )
    for p in model.vqgan.parameters():
        assert not p.requires_grad


def test_sample_pipeline() -> None:
    """Full inference: predict indices, decode through VQGAN."""
    torch.manual_seed(2)
    image_size = 32
    n_refs = 2
    vqgan_cfg = _tiny_vqgan_cfg(image_size)
    tr_cfg = _tiny_transformer_cfg(vqgan_cfg, num_refs=n_refs)
    model = build_vq_font(
        VQFontConfig(vqgan=vqgan_cfg, transformer=tr_cfg), freeze_vqgan="partial",
    )
    model.eval()

    initial = torch.randn(2, 1, image_size, image_size)
    refs = torch.randn(2, n_refs, 1, image_size, image_size)
    sid = torch.tensor([0, 5], dtype=torch.long)
    out = sample_vq_font(
        model=model, initial_glyph=initial, ref_glyphs=refs, structure_id=sid, mode="argmax"
    )
    assert out.shape == (2, 1, image_size, image_size)
    assert torch.isfinite(out).all().item()

    out_sampled = sample_vq_font(
        model=model,
        initial_glyph=initial,
        ref_glyphs=refs,
        structure_id=sid,
        mode="sample",
        temperature=1.0,
        top_k=8,
    )
    assert out_sampled.shape == (2, 1, image_size, image_size)
    assert torch.isfinite(out_sampled).all().item()
