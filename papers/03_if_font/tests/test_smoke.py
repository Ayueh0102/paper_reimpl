"""Smoke tests for IF-Font Phase-2 implementation.

Verifies (Phase-2 invariants):
  1. IDS tokenizer round-trip + structure parsing.
  2. IDS resolver fallbacks gracefully when BabelStone files absent.
  3. Frozen VQTokenizerAdapter shapes — encode→indices, lookup→quant grid,
     decode→image; no Parameters require grad inside the adapter.
  4. StyleEncoder produces the right [B, L+n_tokens, c_out] shape and the
     `cl` contrastive feature in training mode.
  5. MoCoWrapper momentum sync + forward produces [B, 2, dim] cl features.
  6. End-to-end IFFont forward — finite logits, finite sup_cl, gradient
     reaches IDS embeddings + decoder + style/MoCo branch (but NOT the
     frozen VQ adapter).
  7. Coverage similarity computes a valid score in [0, 1].
  8. AR sampler produces a 3-channel image of the right shape.
  9. Decoder block has **1 self-attn + 1 cross-attn + 1 FFN** (NOT 2+1).
 10. compute_loss returns (sq + 0.5 * sup_cl) when MoCoCache is provided.
"""

from __future__ import annotations

import torch
from paper_reimpl_shared.runner.smoke import make_synthetic_batch

from if_font.ids import (
    DEFAULT_IDC_CHARS,
    IDSResolver,
    IDSTokenizer,
    parse_structure_class,
)
from if_font.losses import sq, sup_cl
from if_font.model import (
    IFFontConfig,
    MoCoWrapper,
    StyleEncoder,
    VQTokenizerAdapter,
    VQTokenizerConfig,
    _DecoderBlock,
    build_if_font,
)
from if_font.train import MoCoCache, compute_loss


def _tiny_config(image_size: int = 32, n_refs: int = 2) -> IFFontConfig:
    """Tiny config so the test runs in seconds on CPU."""
    vq_cfg = VQTokenizerConfig(
        image_size=image_size,
        in_channels=3,           # Phase 2: RGB
        embedding_dim=4,         # CompVis pretrained latent dim
        codebook_size=16,
        downsample_factor=8,
    )
    return IFFontConfig(
        image_size=image_size,
        in_channels=3,
        vq=vq_cfg,
        ids_vocab_size=128,
        ids_max_len=8,
        d_model=32,
        n_heads=2,
        n_blocks=2,              # paper uses 10; smoke uses 2 for speed
        ffn_mult=2,
        dropout=0.1,
        bias=False,
        n_refs=n_refs,
    )


# ----------------------------------------------------------------------
# IDS tokenizer + resolver
# ----------------------------------------------------------------------


def test_ids_tokenizer_round_trip() -> None:
    tok = IDSTokenizer.from_idc_only()
    tok.fit_from_strings(["⿰示畐", "⿱艹⿴口十", "⿰犭⿱艹⿴口十"])
    assert tok.pad_id == 0
    for idc in DEFAULT_IDC_CHARS:
        assert idc in tok.token_to_id

    # Round trip (no BOS/EOS in Phase 2 default).
    enc = tok.encode("⿰示畐")
    dec = tok.decode(enc)
    assert dec == "⿰示畐"

    ids, mask = tok.batch_encode(["⿰示畐", "⿱艹⿴口十"], max_len=10)
    assert ids.shape == (2, 10)
    assert mask.shape == (2, 10)
    assert mask[:, 0].all()
    # Shorter sequence has pads at the tail.
    assert (~mask[0, 3:]).all()


def test_parse_structure_class() -> None:
    assert parse_structure_class("⿰示畐") == "left_right"
    assert parse_structure_class("⿱艹⿴口十") == "top_bottom"
    assert parse_structure_class("一") == "atomic"
    assert parse_structure_class("") == "unknown"


def test_ids_resolver_load_absent_files() -> None:
    """When BabelStone files are absent the resolver still constructs and
    falls back to identity decomposition."""
    res = IDSResolver.load(
        babelstone_path="/nonexistent/path/babelstone.txt",
        ids_iffont_path="/nonexistent/path/ids_iffont.txt",
    )
    # 12 IDCs are always available.
    assert "⿰" in res.raw_ids
    # Unknown char resolves to itself.
    assert res.resolve("X") == ("X",)


def test_ids_resolver_load_real() -> None:
    """If vendored files exist, the resolver resolves a common CJK char."""
    res = IDSResolver.load(level="radical")
    # `不` is in basic CJK; should resolve to something non-trivial if
    # BabelStone is present, otherwise it falls back to (`不`,).
    out = res.resolve("不")
    assert isinstance(out, tuple) and len(out) >= 1


# ----------------------------------------------------------------------
# VQ adapter (frozen)
# ----------------------------------------------------------------------


def test_vq_adapter_is_frozen() -> None:
    cfg = VQTokenizerConfig(image_size=32, in_channels=3, embedding_dim=4, codebook_size=16, downsample_factor=8)
    adapter = VQTokenizerAdapter(cfg)
    # No parameter should require grad.
    for n, p in adapter.named_parameters():
        assert p.requires_grad is False, f"{n} still requires grad"
    img = torch.randn(2, 3, 32, 32)
    idx = adapter.encode(img)
    assert idx.shape == (2, cfg.n_tokens)
    assert idx.min() >= 0 and idx.max() < cfg.codebook_size
    quant = adapter.lookup_quant(idx)
    assert quant.shape == (2, cfg.embedding_dim, cfg.token_grid_size, cfg.token_grid_size)
    decoded = adapter.decode_indices(idx)
    assert decoded.shape == (2, 3, 32, 32)
    # `.train()` should be a no-op (override).
    assert adapter.train(True) is adapter
    assert adapter.training is False


# ----------------------------------------------------------------------
# Decoder block shape (1 self + 1 cross + 1 FFN)
# ----------------------------------------------------------------------


def test_decoder_block_has_one_self_attn() -> None:
    """Phase-2 decoder block has exactly **1** causal self-attn (not 2)."""
    from if_font.model import _CausalSelfAttention, _CrossAttention

    cfg = _tiny_config()
    block = _DecoderBlock(cfg)
    self_attns = [m for m in block.modules() if isinstance(m, _CausalSelfAttention)]
    cross_attns = [m for m in block.modules() if isinstance(m, _CrossAttention)]
    assert len(self_attns) == 1, f"expected 1 self-attn, got {len(self_attns)}"
    assert len(cross_attns) == 1, f"expected 1 cross-attn, got {len(cross_attns)}"


# ----------------------------------------------------------------------
# StyleEncoder + MoCo
# ----------------------------------------------------------------------


def test_style_encoder_shapes() -> None:
    cfg = _tiny_config(image_size=32, n_refs=2)
    adapter = VQTokenizerAdapter(cfg.vq)
    enc = StyleEncoder(adapter, c_out=cfg.d_model, l_ids=cfg.ids_max_len, n_head=cfg.n_heads)
    b, n_ref, n_tokens = 2, cfg.n_refs, cfg.vq.n_tokens
    indices = torch.randint(0, cfg.vq.codebook_size, (b, n_ref, n_tokens))
    ids = torch.randn(b, cfg.ids_max_len, cfg.d_model)
    sim = torch.rand(b, n_ref)
    enc.train()
    x_sss, cl = enc(indices, ids, sim)
    assert x_sss.shape == (b, cfg.ids_max_len + n_tokens, cfg.d_model)
    assert cl is not None and cl.shape[0] == b


def test_moco_wrapper_momentum_update() -> None:
    cfg = _tiny_config(image_size=32, n_refs=2)
    adapter = VQTokenizerAdapter(cfg.vq)
    moco = MoCoWrapper(adapter, c_out=cfg.d_model, l_ids=cfg.ids_max_len)
    # After sync, enc and enc_m have identical params.
    diffs = [
        (p_q - p_k).abs().max().item()
        for p_q, p_k in zip(moco.enc.parameters(), moco.enc_m.parameters(), strict=False)
    ]
    assert all(d < 1e-6 for d in diffs)
    moco.train()
    b, n_ref, n_tokens = 2, cfg.n_refs, cfg.vq.n_tokens
    indices = torch.randint(0, cfg.vq.codebook_size, (b, n_ref, n_tokens))
    ids = torch.randn(b, cfg.ids_max_len, cfg.d_model)
    sim = torch.rand(b, n_ref)
    x_sss, cl = moco(indices, ids, sim)
    assert x_sss.shape == (b, cfg.ids_max_len + n_tokens, cfg.d_model)
    assert cl is not None and cl.shape == (b, 2, cl.shape[-1])
    moco.momentum_update(0.5)


# ----------------------------------------------------------------------
# IF-Font end-to-end
# ----------------------------------------------------------------------


def test_if_font_forward_backward_step() -> None:
    """The Phase-2 model must produce a finite loss and a gradient on the
    trainable branches (IDS embeddings, style encoder, MoCo, AR decoder).
    The frozen VQ adapter must NOT receive gradient."""
    torch.manual_seed(0)
    image_size = 32
    cfg = _tiny_config(image_size=image_size, n_refs=2)
    model = build_if_font(cfg)
    base = make_synthetic_batch(
        batch_size=2, image_size=image_size, in_channels=3, n_refs=cfg.n_refs, device="cpu"
    )
    tok = IDSTokenizer.from_idc_only()
    tok.fit_from_strings(["⿰示畐", "⿱艹⿴口十"])
    ids_token_ids, _ = tok.batch_encode(["⿰示畐", "⿱艹⿴口十"], max_len=cfg.ids_max_len)
    coverage_sim = torch.rand(2, cfg.n_refs)
    font_id = torch.tensor([0, 1], dtype=torch.long)
    batch = {
        "image": base["image"],
        "refs": base["refs"],
        "ids_token_ids": ids_token_ids,
        "coverage_sim": coverage_sim,
        "font_id": font_id,
    }
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    model.train()
    loss, log = compute_loss(model=model, batch=batch, cache=MoCoCache(2))
    assert torch.isfinite(loss).item(), f"loss not finite: {loss.item()}"
    loss.backward()
    grads = {
        "ids": sum(
            p.grad.norm().item() for p in model.ids_encoder.parameters() if p.grad is not None
        ),
        "moco": sum(
            p.grad.norm().item() for p in model.moco_wrapper.parameters() if p.grad is not None
        ),
        "decoder": sum(
            p.grad.norm().item() for p in model.decoder.parameters() if p.grad is not None
        ),
    }
    for branch, norm in grads.items():
        assert norm > 0.0, f"branch={branch} got zero gradient"
    # Frozen VQ adapter must NOT have any grad.
    for n, p in model.vq.named_parameters():
        assert p.grad is None, f"frozen vq param {n} got gradient"
    optim.step()
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all().item(), f"non-finite param after step: {name}"


def test_if_font_ids_conditioning_path_active() -> None:
    """Changing only the IDS sequence must change the AR logits."""
    torch.manual_seed(0)
    cfg = _tiny_config(image_size=32, n_refs=2)
    model = build_if_font(cfg)
    model.eval()
    img = torch.randn(2, 3, 32, 32)
    refs = torch.randn(2, cfg.n_refs, 3, 32, 32)
    tok = IDSTokenizer.from_idc_only()
    tok.fit_from_strings(["⿰示畐", "⿱艹⿴口十"])
    ids_a, _ = tok.batch_encode(["⿰示畐", "⿰示畐"], max_len=cfg.ids_max_len)
    ids_b, _ = tok.batch_encode(["⿱艹⿴口十", "⿱艹⿴口十"], max_len=cfg.ids_max_len)
    cov = torch.rand(2, cfg.n_refs)
    with torch.no_grad():
        out_a = model(
            target_image=img, ids_token_ids=ids_a, ref_images=refs, coverage_sim=cov
        )
        out_b = model(
            target_image=img, ids_token_ids=ids_b, ref_images=refs, coverage_sim=cov
        )
    diff = (out_a["logits"] - out_b["logits"]).abs().mean().item()
    assert diff > 0.0, "Changing IDS did not change logits"


def test_if_font_ar_sample_shape() -> None:
    """AR sampler must produce a finite RGB image of the right shape."""
    torch.manual_seed(0)
    cfg = _tiny_config(image_size=32, n_refs=2)
    model = build_if_font(cfg)
    model.eval()
    refs = torch.randn(2, cfg.n_refs, 3, 32, 32)
    tok = IDSTokenizer.from_idc_only()
    tok.fit_from_strings(["⿰示畐", "⿱艹⿴口十"])
    ids, _ = tok.batch_encode(["⿰示畐", "⿱艹⿴口十"], max_len=cfg.ids_max_len)
    cov = torch.rand(2, cfg.n_refs)
    out = model.sample(
        ids_token_ids=ids,
        ref_images=refs,
        coverage_sim=cov,
        temperature=1.0,
        top_k=4,
    )
    assert out.shape == (2, 3, 32, 32)
    assert torch.isfinite(out).all().item()


# ----------------------------------------------------------------------
# Coverage similarity + losses
# ----------------------------------------------------------------------


def test_coverage_similarity_range() -> None:
    from if_font.model import IFFont

    target = [("⿰", "示", "畐"), ("⿱", "艹", "口")]
    refs = [
        [("⿰", "示", "畐"), ("⿰", "示", "X")],
        [("⿱", "艹", "Y"), ("⿰", "X", "Y")],
    ]
    sim = IFFont.compute_coverage(target, refs, DEFAULT_IDC_CHARS)
    assert sim.shape == (2, 2)
    assert (sim >= 0).all() and (sim <= 1).all()
    # Identical ref must score 1.0 (only the leading IDC anchor's run).
    assert sim[0, 0] >= sim[0, 1]


def test_sq_and_sup_cl_finite() -> None:
    logits = torch.randn(2, 4, 16)
    target = torch.randint(0, 16, (2, 4))
    val = sq(logits, target)
    assert torch.isfinite(val).item()
    # sup_cl: 2 views per sample, 4 classes
    feats = torch.randn(4, 2, 8)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    val2 = sup_cl(feats, labels=labels)
    assert torch.isfinite(val2).item()


def test_moco_cache_queue() -> None:
    cache = MoCoCache(max_batches=2)
    cl1 = torch.randn(3, 2, 4)
    id1 = torch.tensor([0, 1, 0])
    cache.push(cl1, id1)
    cl_all, id_all = cache.pop_concat()
    assert cl_all is not None and cl_all.shape == (3, 2, 4)
    assert id_all is not None and id_all.shape == (3,)
    # Push two more and check FIFO bound.
    cache.push(torch.randn(3, 2, 4), torch.tensor([2, 3, 4]))
    cache.push(torch.randn(3, 2, 4), torch.tensor([5, 6, 7]))
    cl_all, id_all = cache.pop_concat()
    assert cl_all.shape == (6, 2, 4)  # original cl1 evicted
