"""Smoke test for QT-Font — Phase 2 paper-aligned redesign.

Verifies:
  1. Glyph-label extraction produces a 3-class label in {0, 1, 2}.
  2. Sparse octree topology is consistent: parent indices align, every leaf
     descends from a "split" parent at full_depth.
  3. D3PM Q-cumulant rows sum to 1 (no row-sum drift).
  4. Multi-depth predict_logits returns per-depth logits with the right shape
     (2 channels at inner depths, 3 at the leaf).
  5. Loss is finite and back-propagates into every major branch.
  6. After an optimizer step, parameters are still finite.
  7. The sampler returns leaf labels in {0, 1, 2}.
"""

from __future__ import annotations

import numpy as np
import torch

from qt_font import (
    D3PMUniform,
    QTFontConfig,
    build_octree_from_image,
    build_octree_from_labels,
    build_qt_font,
    compute_multi_depth_ce,
    extract_glyph_labels,
)
from qt_font.dataset import SyntheticConfig, build_dataset
from qt_font.sample import sample_image, sample_labels
from qt_font.train import compute_loss

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _tiny_config(image_size: int = 32) -> QTFontConfig:
    """Tiny model so the test stays fast on CPU."""
    # depth=5 → leaf grid 32×32; full_depth=2 → dense [0, 2]; depth_stop=2.
    return QTFontConfig(
        image_size=image_size,
        full_depth=2,
        depth=5,
        depth_stop=2,
        n_states=3,
        channels_per_depth=(3, 16, 16, 16, 16, 16, 16),
        cond_dim=32,
        timesteps=20,
        schedule="cos",
        use_style=True,
        use_content=True,
    )


def _glyph_image(size: int, seed: int = 0) -> torch.Tensor:
    """Reproducible glyph-like image with at least one stroke."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = torch.ones((1, 1, size, size))
    # Vertical and horizontal strokes guarantee non-trivial contours.
    img[:, :, size // 3 : size // 3 + max(2, size // 16), 4 : size - 4] = -1.0
    img[:, :, 4 : size - 4, size // 2 : size // 2 + max(2, size // 16)] = -1.0
    del g
    return img


# --------------------------------------------------------------------------- #
# Octree topology                                                              #
# --------------------------------------------------------------------------- #


def test_extract_glyph_labels_three_classes() -> None:
    img = _glyph_image(32, seed=0).squeeze(0).squeeze(0)
    u8 = ((img + 1.0) * 127.5).round().to(torch.uint8).numpy()
    label = extract_glyph_labels(u8)
    assert label.shape == (32, 32)
    assert set(np.unique(label).tolist()) <= {0, 1, 2}, (
        f"labels must be in {{0,1,2}}, saw {np.unique(label).tolist()}"
    )
    # At least one contour and one skeleton pixel should exist.
    assert (label == 1).sum() > 0, "expected at least one contour pixel"
    assert (label == 2).sum() > 0, "expected at least one skeleton pixel"


def test_octree_topology_is_consistent() -> None:
    img = _glyph_image(32, seed=1)
    octree = build_octree_from_image(img, full_depth=2, depth=5)
    assert octree.batch_size == 1
    assert octree.depth == 5
    assert octree.full_depth == 2
    # Every level present.
    for d in range(2, 6):
        assert d in octree.levels, f"missing depth {d}"
    # full_depth level is dense: 2**2 × 2**2 = 16 nodes per sample.
    assert len(octree.levels[2]) == 16
    # Each non-root node has a valid parent index into the previous level.
    for d in range(3, 6):
        lvl = octree.levels[d]
        parent = lvl.parent
        assert (parent >= 0).all(), f"depth {d} has -1 parent"
        prev = octree.levels[d - 1]
        assert (parent < len(prev)).all(), f"depth {d} parent OOB"
    # Leaf depth carries a 3-class label, inner depths do not.
    leaf = octree.levels[5]
    assert leaf.leaf_label is not None
    assert set(leaf.leaf_label.unique().tolist()) <= {0, 1, 2}
    for d in range(2, 5):
        assert octree.levels[d].leaf_label is None


def test_octree_from_synthetic_labels_round_trip() -> None:
    # Build directly from a label map; ensure leaf labels carry the non-empty
    # cells through correctly. (The leaf level may include "split-by-sibling"
    # background cells whose parent at depth-1 was non-empty — that's the
    # adaptive-sparse invariant: a cell exists at depth d if its parent at
    # depth d-1 had any non-empty grand-descendant.)
    label = torch.zeros((1, 8, 8), dtype=torch.long)
    label[0, 2:6, 3] = 1  # vertical contour
    label[0, 4, 1:7] = 2  # horizontal skeleton
    octree = build_octree_from_labels(label, full_depth=1, depth=3)
    leaf = octree.levels[3]
    # All non-empty cells must be retained, with the right labels.
    rs, cs = torch.where(label[0] != 0)
    for r, c in zip(rs.tolist(), cs.tolist(), strict=True):
        match = ((leaf.xy[:, 0] == r) & (leaf.xy[:, 1] == c)).nonzero(as_tuple=False)
        assert match.numel() == 1, f"non-empty cell ({r},{c}) missing from leaf level"
        idx = match[0, 0].item()
        assert leaf.leaf_label[idx].item() == int(label[0, r, c].item())


# --------------------------------------------------------------------------- #
# D3PM                                                                         #
# --------------------------------------------------------------------------- #


def test_d3pm_q_cumulant_row_sums_to_one() -> None:
    diff = D3PMUniform(n_states=3, timesteps=10, schedule="cos")
    row_sums = diff.Q_cum.sum(dim=-1)  # (T, K)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"Q_cum row sums drift: min={row_sums.min().item()} max={row_sums.max().item()}"
    )


def test_d3pm_q_sample_valid_indices() -> None:
    torch.manual_seed(0)
    diff = D3PMUniform(n_states=3, timesteps=10, schedule="cos")
    x0 = torch.randint(0, 3, (64,), dtype=torch.long)
    t = torch.randint(0, 10, (64,), dtype=torch.long)
    xt = diff.q_sample(x0, t)
    assert xt.shape == x0.shape
    assert xt.min().item() >= 0 and xt.max().item() <= 2


# --------------------------------------------------------------------------- #
# Model forward / backward                                                     #
# --------------------------------------------------------------------------- #


def test_predict_logits_shape() -> None:
    torch.manual_seed(0)
    cfg = _tiny_config(image_size=32)
    model = build_qt_font(cfg)
    img = torch.cat([_glyph_image(32, seed=s) for s in range(2)], dim=0)
    gt_octree = build_octree_from_image(img, full_depth=cfg.full_depth, depth=cfg.depth)
    timesteps = torch.tensor([0.0, 1.0])
    cond = model.encode_conditioning(
        timesteps, content_octree=gt_octree, style_octrees=[gt_octree]
    )
    logits = model.predict_logits(gt_octree, cond)
    # Per-depth output shapes.
    for d in range(cfg.depth_stop, cfg.depth + 1):
        n_nodes = len(gt_octree.levels[d])
        expected_K = cfg.n_states if d == cfg.depth else 2
        assert logits[d].shape == (n_nodes, expected_K), (
            f"depth {d} expected {(n_nodes, expected_K)} got {logits[d].shape}"
        )
        assert torch.isfinite(logits[d]).all()


def test_smoke_forward_backward_step() -> None:
    torch.manual_seed(0)
    image_size = 32
    cfg = _tiny_config(image_size=image_size)
    model = build_qt_font(cfg)
    diffusion = D3PMUniform(n_states=cfg.n_states, timesteps=cfg.timesteps, schedule="cos")

    # Build a minibatch with at least 2 different glyphs.
    ds = build_dataset(
        SyntheticConfig(
            length=2,
            image_size=image_size,
            in_channels=1,
            content_channels=1,
            n_refs=1,
            seed=42,
        )
    )
    samples = list(ds)
    batch = {
        k: torch.stack([s[k] for s in samples], dim=0) for k in samples[0].keys()
    }
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss, log = compute_loss(model=model, diffusion=diffusion, batch=batch)
    assert torch.isfinite(loss).item(), f"loss not finite: {loss.item()}"
    assert log["loss_total"] >= 0.0

    loss.backward()

    # Every major branch should receive a gradient.
    branches = {
        "input_conv": model.input_conv,
        "enc_blocks": model.enc_blocks,
        "mid_block1": model.mid_block1,
        "mid_block2": model.mid_block2,
        "dec_blocks": model.dec_blocks,
        "predict_heads": model.predict_heads,
        "time_embed": model.time_embed,
        "style_enc": model.style_enc,
        "content_enc": model.content_enc,
    }
    for name, mod in branches.items():
        grad_norm = sum(p.grad.norm().item() for p in mod.parameters() if p.grad is not None)
        assert grad_norm > 0.0, f"branch={name} got zero gradient"

    optim.step()
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all().item(), f"non-finite param after step: {name}"


def test_multi_depth_ce_components() -> None:
    """The loss MUST emit a `loss_d{depth}` for every depth in [depth_stop, depth]."""
    torch.manual_seed(1)
    cfg = _tiny_config(image_size=32)
    model = build_qt_font(cfg)
    img = torch.cat([_glyph_image(32, seed=s) for s in range(2)], dim=0)
    gt_octree = build_octree_from_image(img, full_depth=cfg.full_depth, depth=cfg.depth)
    timesteps = torch.zeros(2)
    cond = model.encode_conditioning(timesteps, content_octree=gt_octree)
    logits = model.predict_logits(gt_octree, cond)
    loss, log = compute_multi_depth_ce(logits, gt_octree)
    for d in range(cfg.depth_stop, cfg.depth + 1):
        assert f"loss_d{d}" in log
        assert f"acc_d{d}" in log
    assert torch.isfinite(loss).item()


# --------------------------------------------------------------------------- #
# Sampler                                                                      #
# --------------------------------------------------------------------------- #


def test_sample_labels_three_class_output() -> None:
    torch.manual_seed(2)
    cfg = _tiny_config(image_size=32)
    model = build_qt_font(cfg)
    diffusion = D3PMUniform(n_states=cfg.n_states, timesteps=10, schedule="cos")
    img = torch.cat([_glyph_image(32, seed=s) for s in range(2)], dim=0)
    gt_octree = build_octree_from_image(img, full_depth=cfg.full_depth, depth=cfg.depth)
    timesteps = torch.zeros(2)
    cond = model.encode_conditioning(timesteps, content_octree=gt_octree)
    labels = sample_labels(model, diffusion, gt_octree=gt_octree, cond=cond, gap=2)
    assert labels.dtype == torch.long
    assert labels.min().item() >= 0 and labels.max().item() <= 2


def test_sample_image_shape() -> None:
    torch.manual_seed(3)
    image_size = 32
    cfg = _tiny_config(image_size=image_size)
    model = build_qt_font(cfg)
    diffusion = D3PMUniform(n_states=cfg.n_states, timesteps=10, schedule="cos")
    content = torch.cat([_glyph_image(image_size, seed=s) for s in range(2)], dim=0)
    img = sample_image(model, diffusion, batch_size=2, content=content, gap=2)
    assert img.shape == (2, 1, image_size, image_size)
    assert torch.isfinite(img).all().item()
    assert img.min().item() >= -1.0 - 1e-6
    assert img.max().item() <= 1.0 + 1e-6
