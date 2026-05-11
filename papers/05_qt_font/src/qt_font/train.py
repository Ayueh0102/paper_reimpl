"""QT-Font training entry.

Public surfaces
---------------
* :func:`compute_loss` — D3PM uniform cross-entropy on leaf states.
* :func:`main`         — called by ``paper_reimpl_shared.runner.entrypoint``.

The training loop is intentionally minimal: it builds a model, ticks the
optimizer for ``train_cfg.max_steps`` iterations, logs the loss, optionally
stops after one step on ``--dry-run``. Multi-stage TTF / mid-train / Ernantang
behaviour comes from the YAML configs, not from branching code.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SyntheticConfig, build_dataset
from .model import (
    D3PMUniform,
    QTFontConfig,
    QTFontModel,
    build_qt_font,
    quantize_to_states,
)

# --------------------------------------------------------------------------- #
# Loss                                                                         #
# --------------------------------------------------------------------------- #


def _maybe_drop_condition(
    cond_value: torch.Tensor | None,
    *,
    drop_prob: float,
    null_value: int,
) -> torch.Tensor | None:
    """Classifier-free guidance dropout: stochastically replace ids with null id."""
    if cond_value is None or drop_prob <= 0.0:
        return cond_value
    mask = torch.rand(cond_value.shape[0], device=cond_value.device) < drop_prob
    if not mask.any():
        return cond_value
    out = cond_value.clone()
    out[mask] = null_value
    return out


def compute_loss(
    *,
    model: QTFontModel,
    diffusion: D3PMUniform,
    batch: dict[str, torch.Tensor],
    cfg_drop_prob: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Discrete-diffusion CE loss following D3PM.

    Returns
    -------
    loss : scalar tensor
    log  : dict of scalars
    """
    cfg = model.cfg
    image = batch["image"]
    device = image.device
    B = image.shape[0]

    # Build the clean leaf states x_0 from the *target* image.
    x0_states = quantize_to_states(image, depth=cfg.depth, n_states=cfg.n_states)

    # Sample t and x_t.
    t = diffusion.sample_random_step(B, device=device)
    xt_states = diffusion.q_sample(x0_states, t)

    # CFG dropout on conditioning ids.
    char_id = _maybe_drop_condition(
        batch.get("char_id"), drop_prob=cfg_drop_prob, null_value=model.null_char_id
    )
    writer_id = _maybe_drop_condition(
        batch.get("writer_id"), drop_prob=cfg_drop_prob, null_value=model.null_writer_id
    )
    script_id = _maybe_drop_condition(
        batch.get("script_id"), drop_prob=cfg_drop_prob, null_value=model.null_script_id
    )

    # Predict x_0 logits per leaf.
    logits = model.predict_logits_from_states(
        xt_states,
        t,
        content=batch["content"],
        char_id=char_id,
        writer_id=writer_id,
        script_id=script_id,
        ref_images=batch.get("refs"),
        ref_valid=batch.get("ref_valid"),
    )
    loss = diffusion.loss_x0_ce(logits, x0_states)

    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        acc = (preds == x0_states).float().mean()
    log = {
        "loss_total": float(loss.item()),
        "loss_d3pm_ce": float(loss.item()),
        "leaf_acc": float(acc.item()),
        "mean_t": float(t.float().mean().item()),
    }
    return loss, log


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_qt_config(model_cfg: dict[str, Any], data_cfg: dict[str, Any]) -> QTFontConfig:
    section = model_cfg.get("model", model_cfg)
    return QTFontConfig(
        image_size=int(section.get("image_size", 64)),
        in_channels=int(section.get("in_channels", 1)),
        content_channels=int(
            section.get("content_channels", len(data_cfg.get("content_channels", [1])))
        ),
        ref_channels=int(section.get("ref_channels", 1)),
        depth=int(section.get("depth", 4)),
        n_states=int(section.get("n_states", 8)),
        hidden_dim=int(section.get("hidden_dim", 128)),
        n_layers=int(section.get("n_layers", 3)),
        time_embed_dim=int(section.get("time_embed_dim", 128)),
        style_embed_dim=int(section.get("style_embed_dim", 128)),
        content_embed_dim=int(section.get("content_embed_dim", 128)),
        char_vocab_size=int(data_cfg.get("char_vocab_size", 64)),
        writer_vocab_size=int(data_cfg.get("writer_vocab_size", 24)),
        script_vocab_size=int(data_cfg.get("script_vocab_size", 5)),
        dropout=float(section.get("dropout", 0.0)),
        ref_dropout=float(section.get("ref_dropout", 0.1)),
        timesteps=int(section.get("timesteps", 100)),
    )


def _build_synthetic_loader(
    qt_cfg: QTFontConfig, data_cfg: dict[str, Any], batch_size: int, num_workers: int
) -> DataLoader:
    syn_cfg = SyntheticConfig(
        length=int(data_cfg.get("synthetic_length", 32)),
        image_size=qt_cfg.image_size,
        in_channels=qt_cfg.in_channels,
        content_channels=qt_cfg.content_channels,
        n_refs=int(data_cfg.get("max_refs", 1)),
        char_vocab_size=qt_cfg.char_vocab_size,
        writer_vocab_size=qt_cfg.writer_vocab_size,
        script_vocab_size=qt_cfg.script_vocab_size,
        seed=int(data_cfg.get("seed", 0)),
    )
    ds = build_dataset(syn_cfg)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)


def main(
    args,
    *,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    paths,
) -> int:
    """Entry called from ``paper_reimpl_shared.runner.entrypoint``."""
    seed = int(train_cfg.get("seed", 42))
    _seed_everything(seed)

    qt_cfg = _build_qt_config(model_cfg, data_cfg)
    model = build_qt_font(qt_cfg).to(args.device)
    diff_cfg = train_cfg.get("diffusion", {})
    diffusion = D3PMUniform(
        n_states=qt_cfg.n_states,
        timesteps=int(diff_cfg.get("timesteps", qt_cfg.timesteps)),
        beta_start=float(diff_cfg.get("beta_start", 1e-4)),
        beta_end=float(diff_cfg.get("beta_end", 0.02)),
        device=args.device,
    )

    # Either synthetic (smoke / dry-run) or manifest (real run, not implemented).
    source = str(data_cfg.get("source", "synthetic")).lower()
    if args.synthetic or source == "synthetic":
        loader = _build_synthetic_loader(
            qt_cfg,
            data_cfg,
            batch_size=int(train_cfg.get("batch_size", 2)),
            num_workers=int(train_cfg.get("num_workers", 0)),
        )
    else:  # pragma: no cover - placeholder
        raise NotImplementedError(
            "Manifest-backed dataset is a Phase 2/3 deliverable. Use --synthetic."
        )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        betas=(0.9, 0.999),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    print(f"[qt_font] model params = {sum(p.numel() for p in model.parameters()):,}")
    print(f"[qt_font] qt_cfg = {asdict(qt_cfg)}")
    max_steps = int(train_cfg.get("max_steps", 10))
    log_every = int(train_cfg.get("log_every", 1))
    cfg_drop_prob = float(train_cfg.get("cfg_drop_prob", 0.1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    step = 0
    for batch in loader:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        loss, log = compute_loss(
            model=model,
            diffusion=diffusion,
            batch=batch,
            cfg_drop_prob=cfg_drop_prob,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if step % log_every == 0:
            print(f"[qt_font] step={step} {log}")
        step += 1
        if args.dry_run:
            print("[qt_font] dry-run: stop after 1 step")
            return 0
        if step >= max_steps:
            break

    print(f"[qt_font] training done; total_steps={step}")
    return 0
