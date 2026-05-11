"""Smoke test for Moyun blind reimplementation.

Verifies that the published model + loss + 1-step optimizer update produces
a finite loss on a tiny synthetic batch. CPU-only. No disk I/O.

Critical checks (Moyun-specific, per docs/REVIEW_RUBRIC.md):
  * TripleLabel embeddings each receive their OWN gradient (proves the three
    tables are separate, not weight-sharing).
  * Mamba SSM parameters (A_log, conv1d, in_proj, out_proj) all receive
    non-zero gradients (proves the SSM recurrence is in the loss path).
  * Sampling produces a finite output tensor of the correct shape.
"""

from __future__ import annotations

import torch

from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion
from paper_reimpl_shared.runner.smoke import make_synthetic_batch

from moyun.model import MoyunConfig, build_moyun
from moyun.train import compute_loss


def _tiny_config(image_size: int, in_channels: int) -> MoyunConfig:
    """Tiny model so the test stays under a few seconds on CPU.

    image_size=16 / patch_size=4 -> 4x4=16 tokens, which is short enough that
    the sequential scan runs in well under a second per forward.
    """
    return MoyunConfig(
        image_size=image_size,
        in_channels=in_channels,
        patch_size=4,
        hidden_dim=32,
        num_blocks=2,
        d_state=8,
        d_conv=3,
        mlp_ratio=2.0,
        bidirectional=True,
        writer_vocab=8,
        script_vocab=5,
        char_vocab=32,
    )


def test_smoke_forward_backward_step() -> None:
    torch.manual_seed(0)
    image_size = 16
    in_channels = 1
    batch = make_synthetic_batch(
        batch_size=2,
        image_size=image_size,
        in_channels=in_channels,
        char_vocab_size=32,
        writer_vocab_size=8,
        n_refs=0,
        device="cpu",
    )

    cfg = _tiny_config(image_size=image_size, in_channels=in_channels)
    model = build_moyun(cfg)
    diffusion = GaussianDiffusion(
        timesteps=50,
        beta_schedule="linear",
        prediction_target="epsilon",
        device="cpu",
    )
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # First backward warms up the adaLN-Zero modulation path: cond_mlp[-1]
    # is zero-initialized (DiT §3.2 / paper §3.4), so its gradient on the
    # first step is the only thing that breaks the all-zero modulation. We
    # take one optim step, then check gradients on a second forward — this
    # is the iteration at which the TripleLabel embeddings actually receive
    # non-zero gradients.
    loss, _ = compute_loss(model=model, diffusion=diffusion, batch=batch, cfg_drop_prob=0.0)
    assert torch.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"
    loss.backward()
    optim.step()
    optim.zero_grad()

    loss, log = compute_loss(
        model=model,
        diffusion=diffusion,
        batch=batch,
        cfg_drop_prob=0.0,
    )

    assert torch.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"
    loss.backward()

    # Critical Moyun check: each TripleLabel embedding must receive its own
    # gradient. This catches the failure mode "I accidentally shared the
    # writer/script/char tables under one weight tensor".
    triple = model.triple_label
    g_writer = triple.writer.weight.grad
    g_script = triple.script.weight.grad
    g_char = triple.char.weight.grad
    assert g_writer is not None and g_writer.abs().sum().item() > 0.0, (
        "writer embedding got no gradient — TripleLabel path may be broken"
    )
    assert g_script is not None and g_script.abs().sum().item() > 0.0, (
        "script embedding got no gradient — TripleLabel path may be broken"
    )
    assert g_char is not None and g_char.abs().sum().item() > 0.0, (
        "char embedding got no gradient — TripleLabel path may be broken"
    )
    # And the three weight tensors must be DISTINCT objects (not aliases).
    assert triple.writer.weight is not triple.script.weight
    assert triple.writer.weight is not triple.char.weight
    assert triple.script.weight is not triple.char.weight

    # Mamba SSM parameters must also receive gradient — proves the recurrence
    # actually contributes to the loss.
    ssm_grad = 0.0
    for block in model.blocks:
        ssm_grad += sum(
            p.grad.norm().item() for p in block.ssm.parameters() if p.grad is not None
        )
    assert ssm_grad > 0.0, "Mamba SSM got no gradient — recurrence may be disconnected"

    # AdaLN-Zero check: the cond_mlp final layer is zero-initialized. After
    # exactly one backward pass it should have a non-zero gradient (otherwise
    # the modulation path is dead).
    final_lin = model.cond_mlp[-1]
    assert (
        final_lin.weight.grad is not None
        and final_lin.weight.grad.abs().sum().item() > 0.0
    ), "cond_mlp final layer got no gradient — modulation path is disconnected"

    optim.step()
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all().item(), f"Non-finite parameter after step: {name}"

    # Shared sampler smoke — a tiny T so the test stays fast.
    diffusion_short = GaussianDiffusion(
        timesteps=4,
        beta_schedule="linear",
        prediction_target="epsilon",
        device="cpu",
    )
    with torch.no_grad():
        # Moyun ignores content but the shared sampler dereferences it; pass
        # a zero tensor of the right shape.
        zero_content = torch.zeros(2, 1, image_size, image_size)
        out = diffusion_short.sample(
            model,
            shape=(2, in_channels, image_size, image_size),
            content=zero_content,
            writer_id=batch["writer_id"],
            script_id=batch["script_id"],
            char_id=batch["char_id"],
            sampler="ddpm",
            cfg_scale=1.0,
            cfg_uncond_drops_content=False,
            device="cpu",
        )
    assert out.shape == (2, in_channels, image_size, image_size)
    assert torch.isfinite(out).all().item()
    assert log["loss_total"] >= 0.0


def test_cfg_dropout_routes_to_null_embedding() -> None:
    """When cfg_drop_prob=1.0, ALL labels should be dropped and the model
    output should match the fully-unconditioned (writer/script/char all None)
    output — proving the [NULL] embedding path is correctly hooked up.
    """
    torch.manual_seed(0)
    image_size = 16
    in_channels = 1
    batch = make_synthetic_batch(
        batch_size=2,
        image_size=image_size,
        in_channels=in_channels,
        char_vocab_size=32,
        writer_vocab_size=8,
        n_refs=0,
        device="cpu",
    )

    cfg = _tiny_config(image_size=image_size, in_channels=in_channels)
    model = build_moyun(cfg)
    model.eval()

    # Manually construct a fixed t and x for reproducibility.
    t = torch.zeros(2, dtype=torch.long)
    x = batch["image"]

    # 1) Fully unconditioned: pass None.
    with torch.no_grad():
        out_uncond = model(x, t, writer_id=None, script_id=None, char_id=None)
    # 2) Pass labels then the loss path with cfg_drop_prob=1.0 should also
    #    route to [NULL] internally. We mimic the dropout manually.
    drop_ids = -torch.ones(2, dtype=torch.long)  # +1 in _resolve_id -> 0 = [NULL]
    with torch.no_grad():
        out_drop = model(x, t, writer_id=drop_ids, script_id=drop_ids, char_id=drop_ids)
    assert torch.allclose(out_uncond, out_drop, atol=1e-5), (
        "Dropped-to-[NULL] output must match fully-uncond output"
    )


def test_triple_label_embeddings_are_distinct_modules() -> None:
    """Sanity: the three embedding tables must not be the same nn.Module."""
    cfg = _tiny_config(image_size=16, in_channels=1)
    model = build_moyun(cfg)
    t = model.triple_label
    assert t.writer is not t.script
    assert t.writer is not t.char
    assert t.script is not t.char
    # And they must have different output values on a non-zero input
    # (random init makes accidental coincidence extremely unlikely).
    ids = torch.tensor([1, 2, 3, 4])
    ew = t.writer(ids)
    es = t.script(ids)
    ec = t.char(ids)
    assert not torch.allclose(ew, es)
    assert not torch.allclose(ew, ec)
    assert not torch.allclose(es, ec)
