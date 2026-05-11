"""Moyun training — blind reimplementation.

Loss = DDPM denoising MSE (eps-prediction). Paper §3.5 mentions
classifier-free guidance, so during training we per-sample drop the
TripleLabel inputs (independently or jointly) with probability ``cfg_drop_prob``.

Plumbed for the shared entrypoint ``paper_reimpl_shared.runner.entrypoint``::

    main(args, *, data_cfg, model_cfg, train_cfg, paths)
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from paper_reimpl_shared.config import resolve_path
from paper_reimpl_shared.data.legacy import collate_calligraphy_batch
from paper_reimpl_shared.data.manifest import BackendPaths
from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion

from .dataset import build_dataset
from .model import Moyun, MoyunConfig, build_moyun

__all__ = ["compute_loss", "main"]


# --------------------------------------------------------------------------------------
# CFG dropout
# --------------------------------------------------------------------------------------


def _cfg_drop(
    ids: torch.Tensor | None,
    drop_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Replace ids[i] with the [NULL] sentinel for rows where drop_mask is True.

    The model treats real id+1 (so index 0 is reserved for [NULL]). We
    represent [NULL] by setting the row's id to -1 (which becomes index 0
    after ``_resolve_id`` shifts by +1). This is the convention used in
    ``Moyun.forward``.
    """
    if ids is None or drop_mask is None:
        return ids
    out = ids.clone()
    out[drop_mask] = -1  # +1 in _resolve_id -> 0 = [NULL]
    return out


def compute_loss(
    *,
    model: Moyun,
    diffusion: GaussianDiffusion,
    batch: dict[str, torch.Tensor],
    cfg_drop_prob: float = 0.0,
    label_dropout_mode: str = "joint",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the DDPM denoising loss + CFG-dropout TripleLabel.

    Args:
        model: Moyun network.
        diffusion: shared GaussianDiffusion. Prediction target should be
            ``epsilon`` (paper does not specify but DDPM default).
        batch: dict with at least ``image``, ``writer_id``, ``script_id``,
            ``char_id``.
        cfg_drop_prob: probability of dropping a label per sample (Ho &
            Salimans 2022 CFG training trick).
        label_dropout_mode: ``"joint"`` drops all three labels together (one
            Bernoulli per sample); ``"independent"`` drops each label
            independently. Paper does not specify. [guessed] joint is the
            standard CFG recipe.
    """
    x0 = batch["image"]
    writer_id = batch.get("writer_id")
    script_id = batch.get("script_id")
    char_id = batch.get("char_id")

    if cfg_drop_prob > 0.0:
        b = x0.shape[0]
        if label_dropout_mode == "joint":
            drop = torch.rand(b, device=x0.device) < cfg_drop_prob
            writer_id = _cfg_drop(writer_id, drop)
            script_id = _cfg_drop(script_id, drop)
            char_id = _cfg_drop(char_id, drop)
        elif label_dropout_mode == "independent":
            writer_id = _cfg_drop(writer_id, torch.rand(b, device=x0.device) < cfg_drop_prob)
            script_id = _cfg_drop(script_id, torch.rand(b, device=x0.device) < cfg_drop_prob)
            char_id = _cfg_drop(char_id, torch.rand(b, device=x0.device) < cfg_drop_prob)
        else:
            raise ValueError(f"Unknown label_dropout_mode: {label_dropout_mode}")

    diff_batch = diffusion.sample_training_batch(x0)
    model_pred = model(
        diff_batch.x_t,
        diff_batch.timesteps,
        writer_id=writer_id,
        script_id=script_id,
        char_id=char_id,
    )
    loss = F.mse_loss(model_pred, diff_batch.target, reduction="mean")
    log = {"loss_total": float(loss.detach().cpu()), "loss_mse": float(loss.detach().cpu())}
    return loss, log


# --------------------------------------------------------------------------------------
# Config -> dataclass helpers
# --------------------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_cfg_from_yaml(model_cfg: dict[str, Any]) -> MoyunConfig:
    m = model_cfg.get("model", model_cfg)
    return MoyunConfig(
        image_size=int(m.get("image_size", 32)),
        in_channels=int(m.get("in_channels", 4)),
        patch_size=int(m.get("patch_size", 8)),
        hidden_dim=int(m.get("hidden_dim", 512)),
        num_blocks=int(m.get("num_blocks", 4)),
        d_state=int(m.get("d_state", 16)),
        d_conv=int(m.get("d_conv", 3)),
        mlp_ratio=float(m.get("mlp_ratio", 4.0)),
        bidirectional=bool(m.get("bidirectional", True)),
        writer_vocab=int(m.get("writer_vocab", 24)),
        script_vocab=int(m.get("script_vocab", 5)),
        char_vocab=int(m.get("char_vocab", 4659)),
        cond_mlp_dim=m.get("cond_mlp_dim"),
        time_embed_dim=m.get("time_embed_dim"),
    )


def _build_diffusion(train_cfg: dict[str, Any], device: torch.device) -> GaussianDiffusion:
    d = train_cfg.get("diffusion", {})
    return GaussianDiffusion(
        timesteps=int(d.get("timesteps", 1000)),
        beta_start=float(d.get("beta_start", 1e-4)),
        beta_end=float(d.get("beta_end", 2e-2)),
        beta_schedule=str(d.get("beta_schedule", "linear")),
        prediction_target=str(d.get("prediction_target", "epsilon")),
        device=device,
    )


class _CollateNoRefs:
    """Picklable wrapper around ``collate_calligraphy_batch`` with max_refs=0."""

    def __init__(self) -> None:
        self.max_refs = 0

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_calligraphy_batch(batch, max_refs=self.max_refs)


def _build_dataloader(
    *,
    args: argparse.Namespace,
    data_cfg: dict[str, Any],
    model_cfg: MoyunConfig,
    train_cfg: dict[str, Any],
    paths: BackendPaths,
) -> DataLoader:
    dataset = build_dataset(args=args, data_cfg=data_cfg, model_cfg=model_cfg, paths=paths)
    bs = int(train_cfg.get("batch_size", 4))
    nw = int(train_cfg.get("num_workers", 0))
    if getattr(args, "dry_run", False):
        nw = 0
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=False,
        num_workers=nw,
        collate_fn=_CollateNoRefs(),
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def main(
    args: argparse.Namespace,
    *,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    paths: BackendPaths,
) -> int:
    """Entrypoint dispatched from ``paper_reimpl_shared.runner.entrypoint``."""
    device = torch.device(args.device)
    _seed_everything(int(train_cfg.get("seed", 42)))

    cfg = _model_cfg_from_yaml(model_cfg)
    # When using a synthetic dataset (1-channel image), force in_channels=1 so
    # the patchify conv sees the right shape. The YAML default in_channels=4
    # is correct for the "VAE latent" production path.
    if bool(getattr(args, "synthetic", False)):
        cfg.in_channels = 1
    model = build_moyun(cfg).to(device)
    diffusion = _build_diffusion(train_cfg, device=device)

    # Warm-start from --init-ckpt before optimizer build.
    init_ckpt = getattr(args, "init_ckpt", None)
    if init_ckpt:
        blob = torch.load(init_ckpt, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[moyun] warm-start from {init_ckpt} "
            f"(missing={len(missing)} unexpected={len(unexpected)})"
        )

    loader = _build_dataloader(
        args=args, data_cfg=data_cfg, model_cfg=cfg, train_cfg=train_cfg, paths=paths
    )
    lr = float(train_cfg.get("learning_rate", 1e-4))
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    cfg_drop = float(train_cfg.get("cfg_drop_prob", 0.1))
    label_dropout_mode = str(train_cfg.get("label_dropout_mode", "joint"))
    max_steps = int(train_cfg.get("max_steps", 1 if args.dry_run else 1_000_000))
    log_every = int(train_cfg.get("log_every", 50))

    ckpt_dir = train_cfg.get("ckpt_dir")
    if ckpt_dir is not None:
        ckpt_dir = resolve_path(ckpt_dir, base=Path(__file__).resolve().parents[3])
        os.makedirs(ckpt_dir, exist_ok=True)

    print(
        f"[moyun] device={device} steps={max_steps} bs={train_cfg.get('batch_size')} "
        f"lr={lr} cfg_drop={cfg_drop} timesteps={diffusion.timesteps} "
        f"pred={diffusion.prediction_target} schedule={diffusion.beta_schedule} "
        f"hidden_dim={cfg.hidden_dim} num_blocks={cfg.num_blocks} "
        f"patch={cfg.patch_size} in_ch={cfg.in_channels}"
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
                cfg_drop_prob=cfg_drop,
                label_dropout_mode=label_dropout_mode,
            )
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            if step % log_every == 0:
                print(f"[moyun] step={step} loss_total={log['loss_total']:.4f} loss_mse={log['loss_mse']:.4f}")
            step += 1
            if step >= max_steps or args.dry_run:
                break
        if step >= max_steps or args.dry_run:
            break

    if ckpt_dir is not None and not args.dry_run:
        path = Path(ckpt_dir) / "moyun_last.pt"
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, path)
        print(f"[moyun] saved checkpoint -> {path}")

    print(f"[moyun] done; final_step={step} dry_run={args.dry_run}")
    return 0
