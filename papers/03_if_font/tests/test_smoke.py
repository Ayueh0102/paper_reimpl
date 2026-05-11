"""Smoke tests for IF-Font blind reimplementation.

Verifies:
  1. IDS tokenizer round-trip + structure parsing.
  2. VQ tokenizer forward + backward (reconstruction path).
  3. End-to-end IFFont forward + AR cross-entropy backward + optimizer step.
  4. IDS conditioning path actually flows gradient into the IDS encoder.
  5. AR sampler produces a finite image of the right shape.
  6. compute_loss respects the (ce_weight, vq_weight, recon_weight) schedule.
"""

from __future__ import annotations

import torch
from paper_reimpl_shared.runner.smoke import make_synthetic_batch

from if_font.ids import (
    DEFAULT_IDC_CHARS,
    IDSTokenizer,
    parse_structure_class,
)
from if_font.model import (
    IFFontConfig,
    VQTokenizerConfig,
    build_if_font,
)
from if_font.train import compute_loss


def _tiny_config(image_size: int = 32) -> IFFontConfig:
    """Tiny config so the test runs in seconds on CPU."""
    vq_cfg = VQTokenizerConfig(
        image_size=image_size,
        in_channels=1,
        base_channels=16,
        channel_mult=(1, 2, 2, 2),  # 3 downsamples -> factor 8
        embedding_dim=32,
        codebook_size=16,
        commitment_weight=0.25,
        decay=0.99,
    )
    return IFFontConfig(
        image_size=image_size,
        in_channels=1,
        vq=vq_cfg,
        ids_vocab_size=128,
        ids_max_len=8,
        ids_encoder_layers=1,
        ids_encoder_heads=2,
        ids_encoder_dim=32,
        d_model=32,
        n_heads=2,
        n_blocks=2,            # paper uses 10; smoke uses 2 for speed
        n_self_attn_per_block=2,
        ffn_mult=2,
        dropout=0.0,
        n_refs=1,
    )


# ----------------------------------------------------------------------
# IDS tokenizer
# ----------------------------------------------------------------------


def test_ids_tokenizer_round_trip() -> None:
    tok = IDSTokenizer.from_idc_only()
    # Fitting on a real-ish IDS keeps the leaf chars in the vocab.
    tok.fit_from_strings(["⿰示畐", "⿱艹⿴口十", "⿰犭⿱艹⿴口十"])
    # Specials + IDC are already there
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2
    assert tok.unk_id == 3
    for idc in DEFAULT_IDC_CHARS:
        assert idc in tok.token_to_id

    # Round trip
    enc = tok.encode("⿰示畐", add_bos=True, add_eos=True)
    dec = tok.decode(enc)
    assert dec == "⿰示畐"

    # Batch encode with padding
    ids, mask = tok.batch_encode(["⿰示畐", "⿱艹⿴口十"], max_len=10)
    assert ids.shape == (2, 10)
    assert mask.shape == (2, 10)
    # First two cols are BOS + first content token, must be marked as real.
    assert mask[:, 0].all()
    # The shorter sequence (4 chars + BOS + EOS = 6) should have pads at the tail.
    assert (~mask[0, 6:]).all()


def test_parse_structure_class() -> None:
    assert parse_structure_class("⿰示畐") == "left_right"
    assert parse_structure_class("⿱艹⿴口十") == "top_bottom"
    assert parse_structure_class("一") == "atomic"
    assert parse_structure_class("") == "unknown"


# ----------------------------------------------------------------------
# VQ tokenizer
# ----------------------------------------------------------------------


def test_vq_tokenizer_forward_backward() -> None:
    torch.manual_seed(0)
    cfg = _tiny_config(image_size=32)
    model = build_if_font(cfg)
    img = torch.randn(2, 1, 32, 32)
    vq_out = model.vq(img)
    assert vq_out["recon"].shape == img.shape
    assert vq_out["indices"].shape == (2, cfg.vq.n_tokens)
    # Indices are valid codebook entries
    assert int(vq_out["indices"].min()) >= 0
    assert int(vq_out["indices"].max()) < cfg.vq.codebook_size
    # Losses are finite
    assert torch.isfinite(vq_out["vq_loss"]).item()
    assert torch.isfinite(vq_out["recon_loss"]).item()
    # Backward through reconstruction must reach the encoder.
    (vq_out["recon_loss"] + vq_out["vq_loss"]).backward()
    enc_grad = sum(
        p.grad.norm().item() for p in model.vq.encoder.parameters() if p.grad is not None
    )
    dec_grad = sum(
        p.grad.norm().item() for p in model.vq.decoder.parameters() if p.grad is not None
    )
    assert enc_grad > 0.0, "VQ encoder did not receive gradient"
    assert dec_grad > 0.0, "VQ decoder did not receive gradient"


# ----------------------------------------------------------------------
# IF-Font end-to-end
# ----------------------------------------------------------------------


def test_if_font_forward_backward_step() -> None:
    """The full AR + VQ path must produce a finite loss and a gradient
    on every major branch (VQ encoder, IDS encoder, AR decoder)."""
    torch.manual_seed(0)
    image_size = 32
    cfg = _tiny_config(image_size=image_size)
    model = build_if_font(cfg)
    base = make_synthetic_batch(
        batch_size=2, image_size=image_size, in_channels=1, n_refs=1, device="cpu"
    )
    # Build an IDS context that exercises both IDC and leaf tokens.
    tok = IDSTokenizer.from_idc_only()
    tok.fit_from_strings(["⿰示畐", "⿱艹⿴口十"])
    ids_token_ids, ids_attention_mask = tok.batch_encode(
        ["⿰示畐", "⿱艹⿴口十"], max_len=cfg.ids_max_len
    )
    batch = {
        "image": base["image"],
        "refs": base["refs"],
        "ids_token_ids": ids_token_ids,
        "ids_attention_mask": ids_attention_mask,
    }
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss, log = compute_loss(
        model=model,
        batch=batch,
        ce_weight=1.0,
        vq_weight=1.0,
        recon_weight=1.0,
    )
    assert torch.isfinite(loss).item(), f"loss not finite: {loss.item()}"
    loss.backward()
    grad_norms = {
        "vq_encoder": sum(
            p.grad.norm().item() for p in model.vq.encoder.parameters() if p.grad is not None
        ),
        "vq_decoder": sum(
            p.grad.norm().item() for p in model.vq.decoder.parameters() if p.grad is not None
        ),
        "ids_encoder": sum(
            p.grad.norm().item() for p in model.ids_encoder.parameters() if p.grad is not None
        ),
        "ar_decoder": sum(
            p.grad.norm().item() for p in model.decoder.parameters() if p.grad is not None
        ),
    }
    for branch, norm in grad_norms.items():
        assert norm > 0.0, f"branch={branch} got zero gradient — path may be broken"

    optim.step()
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all().item(), f"non-finite parameter after step: {name}"
    assert log["loss_ce"] > 0.0
    assert log["loss_recon"] >= 0.0


def test_if_font_ids_conditioning_path_active() -> None:
    """Changing only the IDS sequence must change the AR logits, proving
    that the IDS branch is actually wired into the decoder."""
    torch.manual_seed(0)
    cfg = _tiny_config(image_size=32)
    model = build_if_font(cfg)
    model.eval()  # disable codebook EMA so two passes are comparable

    img = torch.randn(2, 1, 32, 32)
    refs = torch.randn(2, 1, 1, 32, 32)
    tok = IDSTokenizer.from_idc_only()
    tok.fit_from_strings(["⿰示畐", "⿱艹⿴口十"])
    ids_a, mask_a = tok.batch_encode(["⿰示畐", "⿰示畐"], max_len=cfg.ids_max_len)
    ids_b, mask_b = tok.batch_encode(["⿱艹⿴口十", "⿱艹⿴口十"], max_len=cfg.ids_max_len)
    with torch.no_grad():
        out_a = model(target_image=img, ids_token_ids=ids_a, ids_attention_mask=mask_a, ref_images=refs)
        out_b = model(target_image=img, ids_token_ids=ids_b, ids_attention_mask=mask_b, ref_images=refs)
    # The same image + refs but different IDS must produce different logits.
    diff = (out_a["logits"] - out_b["logits"]).abs().mean().item()
    assert diff > 0.0, "Changing IDS did not change logits — conditioning path disconnected"


def test_if_font_ar_sample_shape() -> None:
    """Autoregressive sampler must produce a finite image of the right shape."""
    torch.manual_seed(0)
    cfg = _tiny_config(image_size=32)
    model = build_if_font(cfg)
    model.eval()
    refs = torch.randn(2, 1, 1, 32, 32)
    tok = IDSTokenizer.from_idc_only()
    tok.fit_from_strings(["⿰示畐", "⿱艹⿴口十"])
    ids, mask = tok.batch_encode(["⿰示畐", "⿱艹⿴口十"], max_len=cfg.ids_max_len)
    out = model.sample(
        ids_token_ids=ids,
        ids_attention_mask=mask,
        ref_images=refs,
        temperature=1.0,
    )
    assert out.shape == (2, 1, 32, 32)
    assert torch.isfinite(out).all().item()


def test_cfg_dropout_does_not_nan() -> None:
    """CFG drop zeroes the IDS mask for some rows. The IDS encoder's
    self-attention used to softmax over an all-masked key set, producing
    NaN logits. Regression: the loss must stay finite for any drop pattern.
    """
    torch.manual_seed(0)
    cfg = _tiny_config(image_size=32)
    model = build_if_font(cfg)
    base = make_synthetic_batch(
        batch_size=4, image_size=32, in_channels=1, n_refs=1, device="cpu"
    )
    tok = IDSTokenizer.from_idc_only()
    tok.fit_from_strings(["⿰示畐", "⿱艹⿴口十"])
    ids, mask = tok.batch_encode(
        ["⿰示畐", "⿰示畐", "⿱艹⿴口十", "⿱艹⿴口十"], max_len=cfg.ids_max_len
    )
    # Manually zero the IDS mask for rows 0 and 2 to simulate CFG drop hitting them.
    mask[0, :] = False
    mask[2, :] = False
    batch = {
        "image": base["image"],
        "refs": base["refs"],
        "ids_token_ids": ids,
        "ids_attention_mask": mask,
    }
    loss, log = compute_loss(
        model=model, batch=batch, ce_weight=1.0, vq_weight=0.0, recon_weight=0.0,
        cfg_drop_prob=0.0,  # already applied by hand above
    )
    assert torch.isfinite(loss).item(), f"loss not finite on CFG drop: {loss.item()}"
    loss.backward()
    # AR decoder + ref VQ encoder must still receive gradient for the
    # dropped rows (since refs are kept).
    ar_grad = sum(p.grad.norm().item() for p in model.decoder.parameters() if p.grad is not None)
    assert ar_grad > 0.0


def test_compute_loss_weight_schedule() -> None:
    """ce_weight=0 → loss collapses to vq + recon (Stage A behaviour)."""
    torch.manual_seed(0)
    cfg = _tiny_config(image_size=32)
    model = build_if_font(cfg)
    base = make_synthetic_batch(
        batch_size=2, image_size=32, in_channels=1, n_refs=1, device="cpu"
    )
    tok = IDSTokenizer.from_idc_only()
    tok.fit_from_strings(["⿰示畐"])
    ids, mask = tok.batch_encode(["⿰示畐", "⿰示畐"], max_len=cfg.ids_max_len)
    batch = {
        "image": base["image"],
        "refs": base["refs"],
        "ids_token_ids": ids,
        "ids_attention_mask": mask,
    }
    loss_a, log_a = compute_loss(
        model=model, batch=batch, ce_weight=0.0, vq_weight=1.0, recon_weight=1.0
    )
    expected = log_a["loss_vq"] + log_a["loss_recon"]
    assert abs(float(loss_a.detach()) - expected) < 1e-5
    assert log_a["loss_ce"] > 0.0  # logged but not contributing
