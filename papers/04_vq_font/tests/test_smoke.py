"""Smoke tests for VQ-Font blind reimplementation.

Three checks:
  1. VQGAN forward + backward + 1 optimizer step (Stage 0) — finite loss,
     finite parameters, codebook indices in [0, K).
  2. Transformer forward + backward + 1 optimizer step (Stage 1+) — finite
     loss, gradient reaches the SSEM structure encoder (conditioning path
     check), VQGAN parameters do NOT receive gradient (frozen check).
  3. Sample path: argmax decode runs without crash and produces an image of
     the expected shape.
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


def _tiny_vqgan_cfg(image_size: int) -> VQGANConfig:
    """Tiny VQGAN: 32px input -> 4x4 latent grid via 3 stages of stride-2."""
    return VQGANConfig(
        image_size=image_size,
        in_channels=1,
        base_channels=16,
        channel_mult=(1, 2, 2),
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
    """Stage 0: train the VQGAN end-to-end on a single tiny batch."""
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

    # All three pieces (encoder, codebook via commitment, decoder) must receive
    # gradient — the straight-through path is what carries it through.
    enc_grad = sum(p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None)
    dec_grad = sum(p.grad.norm().item() for p in model.decoder.parameters() if p.grad is not None)
    cb_grad = model.codebook.codebook.weight.grad.norm().item() if model.codebook.codebook.weight.grad is not None else 0.0
    assert enc_grad > 0.0, "VQGAN encoder received zero gradient"
    assert dec_grad > 0.0, "VQGAN decoder received zero gradient"
    assert cb_grad > 0.0, "VQGAN codebook received zero gradient from VQ losses"

    optim.step()
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all().item(), f"Non-finite parameter after step: {name}"

    # Encode -> indices in [0, K), recon shape matches input.
    indices = model.encode_indices(batch["image"])
    assert indices.dtype == torch.long
    assert indices.min().item() >= 0
    assert indices.max().item() < cfg.num_embeddings
    recon = sample_vqgan_recon(model, batch["image"])
    assert recon.shape == batch["image"].shape
    assert torch.isfinite(recon).all().item()


def test_transformer_stage_smoke() -> None:
    """Stage 1+: train the Transformer with frozen VQGAN."""
    torch.manual_seed(1)
    image_size = 32
    n_refs = 2
    batch = make_synthetic_batch(
        batch_size=2, image_size=image_size, in_channels=1, n_refs=n_refs, device="cpu"
    )
    # The shared synthetic batch produces `refs` instead of `ref_images`; align
    # the key + add the structure_id field this paper needs.
    batch["ref_images"] = batch.pop("refs")
    batch["ref_valid"] = torch.ones(2, n_refs, dtype=torch.bool)
    batch["structure_id"] = torch.tensor([0, 1], dtype=torch.long)

    vqgan_cfg = _tiny_vqgan_cfg(image_size)
    tr_cfg = _tiny_transformer_cfg(vqgan_cfg, num_refs=n_refs)
    model = build_vq_font(VQFontConfig(vqgan=vqgan_cfg, transformer=tr_cfg), freeze_vqgan=True)
    # Verify VQGAN really is frozen.
    for p in model.vqgan.parameters():
        assert not p.requires_grad

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=1e-4)
    loss, log = transformer_compute_loss(model=model, batch=batch, structure_weight=0.1)
    assert torch.isfinite(loss).item(), f"transformer loss not finite: {loss.item()}"
    assert log["loss_token"] > 0.0  # CE on random init should be ~log(K)
    assert log["loss_struct"] > 0.0
    assert 0.0 <= log["token_acc"] <= 1.0
    loss.backward()

    # Conditioning path checks — every branch must see gradient:
    #   transformer.struct_encoder (SSEM additive bias on queries)
    #   transformer cross-attention K/V projections (driven by refs)
    #   transformer token head
    sse_grad = sum(
        p.grad.norm().item()
        for p in model.transformer.struct_encoder.parameters()
        if p.grad is not None
    )
    assert sse_grad > 0.0, "SSEM structure encoder did not receive gradient"
    token_head_grad = sum(
        p.grad.norm().item() for p in model.transformer.token_head.parameters() if p.grad is not None
    )
    assert token_head_grad > 0.0, "token_head did not receive gradient"
    # Cross-attention K/V on first block:
    cross_grad = sum(
        p.grad.norm().item()
        for p in model.transformer.blocks[0].cross_attn.parameters()
        if p.grad is not None
    )
    assert cross_grad > 0.0, "cross_attn did not receive gradient — ref path may be disconnected"
    # VQGAN must NOT have gradient (frozen).
    for name, p in model.vqgan.named_parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0.0, f"frozen VQGAN got gradient: {name}"

    optim.step()
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all().item(), f"Non-finite parameter after step: {name}"


def test_sample_pipeline() -> None:
    """Full inference: predict indices, decode through VQGAN."""
    torch.manual_seed(2)
    image_size = 32
    n_refs = 2
    vqgan_cfg = _tiny_vqgan_cfg(image_size)
    tr_cfg = _tiny_transformer_cfg(vqgan_cfg, num_refs=n_refs)
    model = build_vq_font(VQFontConfig(vqgan=vqgan_cfg, transformer=tr_cfg), freeze_vqgan=True)
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
