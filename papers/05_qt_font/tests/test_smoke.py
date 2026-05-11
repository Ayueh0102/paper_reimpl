"""Smoke test for QT-Font blind reimplementation.

Verifies:
  1. Quadtree construction is deterministic and shape-correct.
  2. D3PM forward q_sample produces valid class indices.
  3. Model.forward (pixel-in / pixel-out adapter) returns a finite image.
  4. compute_loss is finite + back-propagates into every model branch.
  5. After an optimizer step, parameters are still finite.
  6. The discrete sampler returns a correctly-shaped pixel image.
"""

from __future__ import annotations

import torch
from paper_reimpl_shared.runner.smoke import make_synthetic_batch

from qt_font.model import (
    D3PMUniform,
    QTFontConfig,
    build_qt_font,
    build_quadtree_states,
    quantize_to_states,
)
from qt_font.sample import sample_image
from qt_font.train import compute_loss


def _tiny_config(image_size: int = 32, content_channels: int = 1) -> QTFontConfig:
    """Tiny model so the test stays fast on CPU."""
    return QTFontConfig(
        image_size=image_size,
        in_channels=1,
        content_channels=content_channels,
        ref_channels=1,
        depth=3,                # 4^3 = 64 leaves on 8×8 grid
        n_states=4,
        hidden_dim=16,
        n_layers=2,
        time_embed_dim=16,
        style_embed_dim=16,
        content_embed_dim=16,
        char_vocab_size=20,
        writer_vocab_size=8,
        script_vocab_size=5,
        dropout=0.0,
        ref_dropout=0.0,
        timesteps=10,
    )


def test_quadtree_topology_is_deterministic() -> None:
    image = torch.zeros(1, 1, 32, 32)
    states, parent_of, child_of = build_quadtree_states(image, depth=3, n_states=4)
    assert states.shape == (1, 64), "depth=3 should yield 4^3=64 leaves"
    # Full tree node count = (4^4 - 1)/3 = 85
    assert parent_of.shape == (85,)
    assert child_of.shape == (85, 4)
    # Root has no parent.
    assert parent_of[0].item() == -1
    # Last-level nodes have no children.
    assert (child_of[-1] == -1).all().item()
    # Re-running produces the same tensors.
    states2, parent_of2, _ = build_quadtree_states(image, depth=3, n_states=4)
    assert torch.equal(states, states2)
    assert torch.equal(parent_of, parent_of2)


def test_quantize_to_states_value_range() -> None:
    image = torch.linspace(-1.0, 1.0 - 1e-6, 32 * 32).reshape(1, 1, 32, 32)
    states = quantize_to_states(image, depth=3, n_states=4)
    assert states.min().item() >= 0
    assert states.max().item() <= 3


def test_d3pm_q_sample_valid_indices() -> None:
    torch.manual_seed(0)
    diffusion = D3PMUniform(n_states=4, timesteps=10, device="cpu")
    x0 = torch.randint(0, 4, (2, 64), dtype=torch.long)
    t = torch.tensor([0, 9], dtype=torch.long)
    xt = diffusion.q_sample(x0, t)
    assert xt.shape == x0.shape
    assert xt.min().item() >= 0
    assert xt.max().item() <= 3


def test_smoke_forward_backward_step() -> None:
    torch.manual_seed(0)
    image_size = 32
    batch = make_synthetic_batch(
        batch_size=2,
        image_size=image_size,
        in_channels=1,
        char_vocab_size=20,
        writer_vocab_size=8,
        n_refs=1,
        device="cpu",
    )
    # Down-channel the content tensor for our tiny config.
    batch["content"] = batch["content"][:, :1]

    cfg = _tiny_config(image_size=image_size, content_channels=1)
    model = build_qt_font(cfg)
    diffusion = D3PMUniform(n_states=cfg.n_states, timesteps=cfg.timesteps, device="cpu")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss, log = compute_loss(model=model, diffusion=diffusion, batch=batch, cfg_drop_prob=0.0)
    assert torch.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"
    assert log["loss_total"] >= 0.0

    loss.backward()

    # Each major conditioning + processing branch must receive non-zero gradient.
    branches = {
        "state_embed": model.state_embed,
        "content_encoder": model.content_encoder,
        "style_encoder": model.style_encoder,
        "char_embed": model.char_embed,
        "writer_embed": model.writer_embed,
        "fine_layers": model.fine_layers,
        "coarse_layers": model.coarse_layers,
        "pool": model.pool,
        "head": model.head,
    }
    for name, mod in branches.items():
        grad_norm = sum(p.grad.norm().item() for p in mod.parameters() if p.grad is not None)
        # style_encoder is fed by refs which were provided, so it must get gradient.
        assert grad_norm > 0.0, (
            f"branch={name} got zero gradient — conditioning path may be broken"
        )

    optim.step()
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all().item(), f"non-finite param after step: {name}"


def test_pixel_adapter_forward() -> None:
    """The standard ``model.forward(x_t, t, content=..., ...)`` adapter."""
    torch.manual_seed(1)
    image_size = 32
    cfg = _tiny_config(image_size=image_size, content_channels=1)
    model = build_qt_font(cfg)
    x_t = torch.randn(2, 1, image_size, image_size)
    t = torch.tensor([0, 5], dtype=torch.long)
    content = torch.randn(2, 1, image_size, image_size)
    out = model(x_t, t, content=content)
    assert out.shape == (2, 1, image_size, image_size)
    assert torch.isfinite(out).all().item()


def test_sample_image_shape() -> None:
    torch.manual_seed(2)
    image_size = 32
    cfg = _tiny_config(image_size=image_size, content_channels=1)
    model = build_qt_font(cfg)
    diffusion = D3PMUniform(n_states=cfg.n_states, timesteps=4, device="cpu")
    content = torch.randn(2, 1, image_size, image_size)
    img = sample_image(model, diffusion, batch_size=2, content=content)
    assert img.shape == (2, 1, image_size, image_size)
    assert torch.isfinite(img).all().item()
    assert img.min().item() >= -1.0 - 1e-6
    assert img.max().item() <= 1.0 + 1e-6
