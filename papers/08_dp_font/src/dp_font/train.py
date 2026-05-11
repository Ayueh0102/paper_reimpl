"""DP-Font training entry point — blind reimplementation.

Plumbed for the shared entrypoint ``paper_reimpl_shared.runner.entrypoint``:
    ``main(args, *, data_cfg, model_cfg, train_cfg, paths)``.

Loss = L_simple (DDPM denoising on epsilon) + λ_PINN * L_PINN, where
L_PINN combines ink-diffusion residual, nib-motion smoothness, and stroke
continuity (see ``pinn_losses.py``).

DP-Font's classifier-free guidance drops the categorical attributes (writer
/ script / char) with per-attribute Bernoulli probability ``cfg_drop_prob``
at training time; sampling-time ``cfg_scale`` then steers between the
fully-conditional and the fully-null predictions.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from paper_reimpl_shared.config import resolve_path
from paper_reimpl_shared.data.manifest import BackendPaths
from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion

from .dataset import _DPFontCollate, build_dataset
from .model import DPFont, DPFontConfig, build_dp_font
from .pinn_losses import pinn_loss


__all__ = ["compute_loss", "main"]


# ---------------------------------------------------------------------------
# CFG dropout helper
# ---------------------------------------------------------------------------


def _cfg_drop(ids: torch.Tensor | None, *, p: float, null_id: int) -> torch.Tensor | None:
    """Randomly replace categorical ids with ``null_id`` with probability p.

    DP-Font's multi-attribute CFG drops each attribute independently. This
    mirrors Ho & Salimans (2022) but applied per-attribute (one Bernoulli
    coin per sample per attribute).
    """
    if ids is None or p <= 0.0:
        return ids
    drop = torch.rand(ids.shape[0], device=ids.device) < p
    return torch.where(drop, torch.full_like(ids, null_id), ids)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_loss(
    *,
    model: DPFont,
    diffusion: GaussianDiffusion,
    batch: dict[str, torch.Tensor],
    pinn_weight: float,
    pinn_weights: dict[str, float] | None = None,
    cfg_drop_prob: float = 0.0,
    skeleton_channel_index: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute L_simple + λ_PINN * L_PINN.

    Args:
        model: DP-Font model.
        diffusion: shared GaussianDiffusion (epsilon target recommended).
        batch: dict from ``collate_dp_font_batch``.
        pinn_weight: λ_PINN — the master scale for the PINN aggregate.
        pinn_weights: per-term weights inside ``pinn_loss``. Keys:
            ``weight_diffusion``, ``weight_nib``, ``weight_continuity``,
            ``nu``. Defaults all to 1.0.
        cfg_drop_prob: per-attribute null-id replacement probability.
        skeleton_channel_index: which content channel is the skeleton. If
            None (or out-of-range), nib-motion smoothness uses the predicted
            glyph itself.
    """
    x0 = batch["image"]
    content = batch["content"]
    null_ids = model.guidance.null_ids

    writer_id = _cfg_drop(batch.get("writer_id"), p=cfg_drop_prob, null_id=null_ids["writer"])
    script_id = _cfg_drop(batch.get("script_id"), p=cfg_drop_prob, null_id=null_ids["script"])
    char_id = _cfg_drop(batch.get("char_id"), p=cfg_drop_prob, null_id=null_ids["char"])
    stroke_order = batch.get("stroke_order")
    ink_intensity = batch.get("ink_intensity")
    font_size = batch.get("font_size")

    diff_batch = diffusion.sample_training_batch(x0)
    model_pred = model(
        diff_batch.x_t,
        diff_batch.timesteps,
        content=content,
        char_id=char_id,
        script_id=script_id,
        writer_id=writer_id,
        stroke_order=stroke_order,
        ink_intensity=ink_intensity,
        font_size=font_size,
    )
    loss_simple = F.mse_loss(model_pred, diff_batch.target, reduction="mean")

    log: dict[str, float] = {
        "loss_total": 0.0,
        "loss_simple": float(loss_simple.detach().cpu()),
        "loss_pinn": 0.0,
    }

    loss_total = loss_simple
    if pinn_weight > 0.0:
        # Reconstruct predicted x0 from the model output (epsilon-prediction
        # → x0). For x0-prediction the shared helper returns the clamped
        # prediction directly.
        x0_pred = diffusion.predict_x0(diff_batch.x_t, diff_batch.timesteps, model_pred)

        # Extract skeleton channel if available; otherwise None lets the
        # nib-motion term fall back to x0_pred.
        skeleton: torch.Tensor | None = None
        if (
            skeleton_channel_index is not None
            and 0 <= skeleton_channel_index < content.shape[1]
        ):
            skeleton = content[:, skeleton_channel_index : skeleton_channel_index + 1]

        weights = pinn_weights or {}
        pinn_total, pinn_log = pinn_loss(
            x0_pred,
            skeleton=skeleton,
            weight_diffusion=float(weights.get("weight_diffusion", 1.0)),
            weight_nib=float(weights.get("weight_nib", 1.0)),
            weight_continuity=float(weights.get("weight_continuity", 1.0)),
            nu=float(weights.get("nu", 1.0)),
        )
        log.update(pinn_log)
        log["loss_pinn"] = float(pinn_total.detach().cpu())
        loss_total = loss_total + float(pinn_weight) * pinn_total

    log["loss_total"] = float(loss_total.detach().cpu())
    return loss_total, log


# ---------------------------------------------------------------------------
# Trainer wrapper
# ---------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_cfg_from_yaml(model_cfg: dict[str, Any]) -> DPFontConfig:
    m = model_cfg.get("model", model_cfg)
    return DPFontConfig(
        image_size=int(m.get("image_size", 80)),
        in_channels=int(m.get("in_channels", 1)),
        content_channels=int(m.get("content_channels", 1)),
        base_channels=int(m.get("base_channels", 64)),
        channel_mult=tuple(int(x) for x in m.get("channel_mult", [1, 2, 2, 4])),
        attn_resolutions=tuple(int(x) for x in m.get("attn_resolutions", [10])),
        num_res_blocks=int(m.get("num_res_blocks", 2)),
        time_embed_dim=int(m.get("time_embed_dim", 256)),
        cond_embed_dim=int(m.get("cond_embed_dim", 256)),
        num_heads=int(m.get("num_heads", 4)),
        dropout=float(m.get("dropout", 0.0)),
        writer_vocab_size=int(m.get("writer_vocab_size", 32)),
        script_vocab_size=int(m.get("script_vocab_size", 4)),
        char_vocab_size=int(m.get("char_vocab_size", 5000)),
        stroke_vocab_size=int(m.get("stroke_vocab_size", 36)),
        stroke_seq_len=int(m.get("stroke_seq_len", 32)),
        use_ink_intensity=bool(m.get("use_ink_intensity", True)),
        use_font_size=bool(m.get("use_font_size", True)),
    )


def _build_diffusion(train_cfg: dict[str, Any], device: torch.device) -> GaussianDiffusion:
    d = train_cfg.get("diffusion", {})
    return GaussianDiffusion(
        timesteps=int(d.get("timesteps", 1000)),
        beta_start=float(d.get("beta_start", 1e-4)),
        beta_end=float(d.get("beta_end", 2e-2)),
        beta_schedule=str(d.get("beta_schedule", "cosine")),
        prediction_target=str(d.get("prediction_target", "epsilon")),
        device=device,
    )


def _build_dataloader(
    *,
    args,
    data_cfg: dict[str, Any],
    model_cfg: DPFontConfig,
    train_cfg: dict[str, Any],
    paths: BackendPaths,
) -> DataLoader:
    dataset = build_dataset(args=args, data_cfg=data_cfg, model_cfg=model_cfg, paths=paths)
    bs = int(train_cfg.get("batch_size", 4))
    nw = int(train_cfg.get("num_workers", 0))
    if getattr(args, "dry_run", False):
        nw = 0
    max_refs = int(data_cfg.get("max_refs", 0))
    collate = _DPFontCollate(max_refs)
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=False,
        num_workers=nw,
        collate_fn=collate,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def main(args, *, data_cfg, model_cfg, train_cfg, paths: BackendPaths) -> int:
    device = torch.device(args.device)
    _seed_everything(int(train_cfg.get("seed", 42)))

    cfg = _model_cfg_from_yaml(model_cfg)
    model = build_dp_font(cfg).to(device)
    diffusion = _build_diffusion(train_cfg, device=device)

    pinn_weight = float(train_cfg.get("pinn_weight", 0.0))
    pinn_weights = dict(train_cfg.get("pinn_weights", {})) if train_cfg.get("pinn_weights") else None
    skeleton_idx = train_cfg.get("skeleton_channel_index")
    skeleton_idx = int(skeleton_idx) if skeleton_idx is not None else None

    loader = _build_dataloader(
        args=args,
        data_cfg=data_cfg,
        model_cfg=cfg,
        train_cfg=train_cfg,
        paths=paths,
    )
    lr = float(train_cfg.get("learning_rate", 1e-4))
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    cfg_drop = float(train_cfg.get("cfg_drop_prob", 0.0))
    max_steps = int(train_cfg.get("max_steps", 1 if args.dry_run else 1_000_000))
    log_every = int(train_cfg.get("log_every", 50))

    ckpt_dir = train_cfg.get("ckpt_dir")
    if ckpt_dir is not None:
        ckpt_dir = resolve_path(ckpt_dir, base=Path(__file__).resolve().parents[3])
        os.makedirs(ckpt_dir, exist_ok=True)

    print(
        f"[dp_font] device={device} steps={max_steps} bs={train_cfg.get('batch_size')} "
        f"lr={lr} pinn_weight={pinn_weight} cfg_drop={cfg_drop} timesteps={diffusion.timesteps} "
        f"pred={diffusion.prediction_target} schedule={diffusion.beta_schedule} "
        f"img={cfg.image_size}px content_C={cfg.content_channels}"
    )

    model.train()
    step = 0
    for _epoch in range(int(train_cfg.get("max_epochs", 1))):
        for batch in loader:
            batch = _move_batch(batch, device)
            optim.zero_grad(set_to_none=True)
            loss, log = compute_loss(
                model=model,
                diffusion=diffusion,
                batch=batch,
                pinn_weight=pinn_weight,
                pinn_weights=pinn_weights,
                cfg_drop_prob=cfg_drop,
                skeleton_channel_index=skeleton_idx,
            )
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            if step % log_every == 0:
                print(
                    f"[dp_font] step={step} loss_total={log['loss_total']:.4f} "
                    f"loss_simple={log['loss_simple']:.4f} loss_pinn={log['loss_pinn']:.4f}"
                )
            step += 1
            if step >= max_steps or args.dry_run:
                break
        if step >= max_steps or args.dry_run:
            break

    if ckpt_dir is not None and not args.dry_run:
        path = Path(ckpt_dir) / "dp_font_last.pt"
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, path)
        print(f"[dp_font] saved checkpoint -> {path}")

    print(f"[dp_font] done; final_step={step} dry_run={args.dry_run}")
    return 0
