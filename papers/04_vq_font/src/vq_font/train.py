"""VQ-Font training.

Two stages are dispatched by ``train_cfg.stage``:

* ``stage: vqgan`` (Stage 0)
    Trains the VQGAN font codebook end-to-end with the official
    ``VQLPIPSWithDiscriminator`` recipe::

        L_G = nll(L1 + λ_perc · LPIPS)
            + adopt_weight(d_weight · disc_factor, step >= disc_start) · g_loss
            + codebook_weight · L_vq

        L_D = adopt_weight(disc_factor, step >= disc_start) · hinge_D(real, fake)

    Defaults (``vqgan/custom_vqgan.yaml`` of the official repo):
      ``disc_start=10000, disc_weight=0.8, codebook_weight=1.0,
      perceptual_weight=1.0, disc_in_channels=1``. Adam(β=(0.5, 0.9))
      for both generator and discriminator.

* ``stage: transformer`` (Stage 1+, paper-cited 1.5M iters)
    Trains the Token Prior Refinement Transformer with cross-entropy on
    the target codebook indices. SSEM is now the parameter-free
    ``RegionAttentionRecalibrator`` inside the transformer — there is no
    auxiliary structure CE term anymore. Optimizer: Adam(β=(0.0, 0.9))
    with ``g_lr=2e-4``, StepLR(step_size=10000, gamma=0.95).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from paper_reimpl_shared.config import resolve_path
from paper_reimpl_shared.data.manifest import BackendPaths

from .dataset import VQFontCollate, build_dataset
from .model import (
    VQFont,
    VQFontConfig,
    VQGAN,
    VQGANConfig,
    TransformerConfig,
    build_vq_font,
    build_vqgan,
)
from .vqgan_loss import VQLPIPSLossConfig, VQLPIPSWithDiscriminator

__all__ = [
    "vqgan_compute_loss",
    "transformer_compute_loss",
    "main",
]


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Config builders
# --------------------------------------------------------------------------------------


def _vqgan_cfg_from_yaml(model_cfg: dict[str, Any]) -> VQGANConfig:
    """Read either model_cfg['vqgan'] or model_cfg root keys."""
    raw = model_cfg.get("vqgan", model_cfg)
    return VQGANConfig(
        image_size=int(raw.get("image_size", 128)),
        in_channels=int(raw.get("in_channels", 1)),
        base_channels=int(raw.get("base_channels", 32)),
        channel_mult=tuple(int(x) for x in raw.get("channel_mult", [1, 1, 2, 4])),
        z_channels=int(raw.get("z_channels", 256)),
        embed_dim=int(raw.get("embed_dim", 256)),
        num_embeddings=int(raw.get("num_embeddings", 1024)),
        commitment_weight=float(raw.get("commitment_weight", 0.25)),
        num_res_blocks=int(raw.get("num_res_blocks", 3)),
        dropout=float(raw.get("dropout", 0.0)),
    )


def _transformer_cfg_from_yaml(
    model_cfg: dict[str, Any], vqgan_cfg: VQGANConfig
) -> TransformerConfig:
    raw = model_cfg.get("transformer", {})
    return TransformerConfig(
        image_size=vqgan_cfg.image_size,
        latent_resolution=int(raw.get("latent_resolution", vqgan_cfg.out_resolution())),
        embed_dim=int(raw.get("embed_dim", vqgan_cfg.embed_dim)),
        num_blocks=int(raw.get("num_blocks", 15)),
        num_heads=int(raw.get("num_heads", 8)),
        mlp_ratio=float(raw.get("mlp_ratio", 2.0)),
        dropout=float(raw.get("dropout", 0.0)),
        num_refs=int(raw.get("num_refs", 3)),
        codebook_size=int(raw.get("codebook_size", vqgan_cfg.num_embeddings)),
        num_structures=int(raw.get("num_structures", 13)),
    )


def _vqgan_loss_cfg_from_yaml(train_cfg: dict[str, Any]) -> VQLPIPSLossConfig:
    """Build the VQLPIPS loss config from the train yaml.

    Falls back to the official defaults. ``train_cfg.loss`` can override
    any field, e.g.::

        loss:
          disc_start: 10000
          disc_weight: 0.8
          codebook_weight: 1.0
    """
    raw = dict(train_cfg.get("loss", {}))
    return VQLPIPSLossConfig(
        disc_start=int(raw.get("disc_start", 10000)),
        codebook_weight=float(raw.get("codebook_weight", 1.0)),
        pixelloss_weight=float(raw.get("pixelloss_weight", 1.0)),
        perceptual_weight=float(raw.get("perceptual_weight", 1.0)),
        disc_num_layers=int(raw.get("disc_num_layers", 3)),
        disc_in_channels=int(raw.get("disc_in_channels", 1)),
        disc_factor=float(raw.get("disc_factor", 1.0)),
        disc_weight=float(raw.get("disc_weight", 0.8)),
        disc_ndf=int(raw.get("disc_ndf", 64)),
        disc_loss=str(raw.get("disc_loss", "hinge")),
    )


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------


def vqgan_compute_loss(
    *,
    model: VQGAN,
    batch: dict[str, torch.Tensor],
    recon_weight: float = 1.0,
    vq_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pure L1 + commitment loss (legacy helper, kept for smoke tests).

    The full Stage 0 trainer uses :class:`VQLPIPSWithDiscriminator` directly;
    this helper preserves the simpler single-loss surface the smoke test
    relies on (no discriminator weights, no LPIPS download).
    """
    x = batch["image"]
    out = model(x)
    loss_recon = F.l1_loss(out.recon, x, reduction="mean")
    loss_vq = out.vq_loss
    total = recon_weight * loss_recon + vq_weight * loss_vq
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_recon": float(loss_recon.detach().cpu()),
        "loss_vq": float(loss_vq.detach().cpu()),
    }


def transformer_compute_loss(
    *,
    model: VQFont,
    batch: dict[str, torch.Tensor],
    **_kwargs: Any,  # accepts legacy ``structure_weight`` kwarg silently.
) -> tuple[torch.Tensor, dict[str, float]]:
    """L_token_ce only.

    SSEM is parameter-free (RegionAttentionRecalibrator inside the
    transformer); there is no aux structure-CE term in the paper-faithful
    setup. ``structure_weight`` is accepted for backward compatibility but
    ignored.
    """
    target = batch["image"]
    # Phase 1 stand-in: use the target's content channel (source glyph render)
    # as the initial synthesis. Real pipeline would call an FFG module here.
    initial = batch.get("content")
    if initial is None or initial.numel() == 0:
        initial = target
    if initial.shape[1] != target.shape[1]:
        initial = initial[:, : target.shape[1]]
    refs = batch["ref_images"]
    structure_id = batch.get("structure_id")
    if structure_id is None:
        # Default to class 1 ('atomic' / full-map template) when SSEM ids
        # are missing — see dataset.py STRUCTURE_NAME_TO_ID rationale.
        structure_id = torch.full(
            (target.shape[0],), 1, dtype=torch.long, device=target.device
        )
    ref_valid = batch.get("ref_valid")

    token_logits, _attn_map = model.predict_token_logits(
        initial, refs, structure_id, ref_valid=ref_valid
    )

    with torch.no_grad():
        target_indices = model.encode_target_indices(target)  # [B, H_lat, W_lat]
    target_flat = target_indices.reshape(target.shape[0], -1)
    b, n, k = token_logits.shape
    loss_token = F.cross_entropy(
        token_logits.reshape(b * n, k),
        target_flat.reshape(b * n),
        reduction="mean",
    )
    total = loss_token
    with torch.no_grad():
        pred = token_logits.argmax(dim=-1)
        token_acc = (pred == target_flat).float().mean().item()
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_token": float(loss_token.detach().cpu()),
        "token_acc": float(token_acc),
    }


# --------------------------------------------------------------------------------------
# Training plumbing
# --------------------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _build_dataloader(
    *,
    args: argparse.Namespace,
    data_cfg: dict[str, Any],
    model_cfg: VQFontConfig | VQGANConfig,
    train_cfg: dict[str, Any],
    paths: BackendPaths,
) -> DataLoader:
    dataset = build_dataset(args=args, data_cfg=data_cfg, model_cfg=model_cfg, paths=paths)
    bs = int(train_cfg.get("batch_size", 4))
    nw = int(train_cfg.get("num_workers", 0))
    if getattr(args, "dry_run", False):
        nw = 0
    max_refs = int(data_cfg.get("max_refs", 3))
    collate = VQFontCollate(max_refs)
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=False,
        num_workers=nw,
        collate_fn=collate,
    )


def _save_state_with_cfg(
    *,
    state: dict[str, Any],
    cfg_dict: dict[str, Any],
    ckpt_path: Path,
) -> None:
    """Persist a state-dict + JSON sidecar config side-by-side."""
    torch.save(state, ckpt_path)
    cfg_path = ckpt_path.with_suffix(ckpt_path.suffix + ".cfg.json")
    cfg_path.write_text(json.dumps(cfg_dict, sort_keys=True, indent=2))


def _load_vqgan_ckpt(
    model: VQFont | VQGAN, ckpt_path: str | Path, *, strict: bool = True
) -> None:
    """Load a VQGAN state-dict checkpoint with ``weights_only=True``."""
    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001
        logger.warning(
            "[vq_font] checkpoint %s is legacy (weights_only=False fallback). "
            "Re-save via training to get the safe split-config format.",
            ckpt_path,
        )
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    if not isinstance(state, dict):
        raise TypeError(
            f"VQGAN checkpoint at {ckpt_path} did not yield a state-dict "
            f"(got {type(state).__name__})."
        )
    if isinstance(model, VQFont):
        model.vqgan.load_state_dict(state, strict=strict)
    else:
        model.load_state_dict(state, strict=strict)


def _parse_adam_betas(train_cfg: dict[str, Any], default: tuple[float, float]) -> tuple[float, float]:
    """Read ``adam_betas`` from train_cfg, falling back to ``default``."""
    raw = train_cfg.get("adam_betas", default)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    return default


def _get_last_layer(model: VQGAN) -> torch.Tensor:
    """Return the last decoder conv weight, for adaptive disc weight.

    Mirrors ``VQModel.get_last_layer``. With our InstanceNorm conv stack
    the final pre-Tanh conv is ``decoder.out_conv.conv.weight``.
    """
    return model.decoder.out_conv.conv.weight


def _run_vqgan_stage(
    args: argparse.Namespace,
    *,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    paths: BackendPaths,
) -> int:
    device = torch.device(args.device)
    _seed_everything(int(train_cfg.get("seed", 42)))
    vqgan_cfg = _vqgan_cfg_from_yaml(model_cfg)
    model = build_vqgan(vqgan_cfg).to(device)

    # Warm-start from --init-ckpt before optimizer build.
    init_ckpt = getattr(args, "init_ckpt", None)
    if init_ckpt:
        blob = torch.load(init_ckpt, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            "[vq_font/vqgan] warm-start from %s (missing=%d unexpected=%d)",
            init_ckpt,
            len(missing),
            len(unexpected),
        )

    # Build loss (LPIPS + Discriminator). ``simple_loss=True`` falls back to
    # pure L1 + commitment for dry-runs / smoke (no LPIPS download).
    simple_loss = bool(train_cfg.get("simple_loss", False)) or args.dry_run
    if simple_loss:
        loss_module: VQLPIPSWithDiscriminator | None = None
        logger.info("[vq_font/vqgan] simple_loss=True — using L1 + commitment only")
    else:
        loss_module = VQLPIPSWithDiscriminator(_vqgan_loss_cfg_from_yaml(train_cfg)).to(device)

    loader = _build_dataloader(
        args=args, data_cfg=data_cfg, model_cfg=vqgan_cfg, train_cfg=train_cfg, paths=paths
    )

    # Two-optimizer split (matches ``VQModel.configure_optimizers``):
    # G optimizes encoder + decoder + quantize + pre_quant + post_quant;
    # D optimizes the NLayerDiscriminator. Both use Adam(betas=(0.5, 0.9)).
    g_lr = float(train_cfg.get("g_lr", train_cfg.get("learning_rate", 4.5e-6)))
    d_lr = float(train_cfg.get("d_lr", g_lr))
    betas = _parse_adam_betas(train_cfg, default=(0.5, 0.9))
    g_params = (
        list(model.encoder.parameters())
        + list(model.decoder.parameters())
        + list(model.codebook.parameters())
        + (list(model.pre_quant.parameters()) if isinstance(model.pre_quant, torch.nn.Module)
           and any(True for _ in model.pre_quant.parameters()) else [])
        + (list(model.post_quant.parameters()) if isinstance(model.post_quant, torch.nn.Module)
           and any(True for _ in model.post_quant.parameters()) else [])
    )
    optim_g = torch.optim.Adam(g_params, lr=g_lr, betas=betas,
                               weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    if loss_module is not None:
        optim_d = torch.optim.Adam(loss_module.discriminator.parameters(),
                                   lr=d_lr, betas=betas,
                                   weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    else:
        optim_d = None

    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    recon_w = float(train_cfg.get("recon_weight", 1.0))
    vq_w = float(train_cfg.get("vq_weight", 1.0))
    max_steps = int(train_cfg.get("max_steps", 1 if args.dry_run else 200_000))
    log_every = int(train_cfg.get("log_every", 100))
    ckpt_dir = train_cfg.get("ckpt_dir")
    if ckpt_dir is not None:
        ckpt_dir = resolve_path(ckpt_dir, base=Path(__file__).resolve().parents[3])
        os.makedirs(ckpt_dir, exist_ok=True)

    logger.info(
        "[vq_font/vqgan] device=%s bs=%s g_lr=%s d_lr=%s betas=%s steps=%d K=%d z_grid=%d",
        device, train_cfg.get("batch_size"), g_lr, d_lr, betas, max_steps,
        vqgan_cfg.num_embeddings, vqgan_cfg.out_resolution(),
    )

    model.train()
    step = 0
    for _epoch in range(int(train_cfg.get("max_epochs", 1))):
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            x = batch["image"]

            if loss_module is None:
                # Simple-loss path (L1 + commitment).
                optim_g.zero_grad(set_to_none=True)
                loss, log = vqgan_compute_loss(
                    model=model, batch=batch, recon_weight=recon_w, vq_weight=vq_w
                )
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim_g.step()
                if step % log_every == 0:
                    logger.info(
                        "[vq_font/vqgan] step=%d loss=%.4f recon=%.4f vq=%.4f",
                        step, log["loss_total"], log["loss_recon"], log["loss_vq"],
                    )
            else:
                # Full loss path: G then D pass.
                out = model(x)
                # Generator branch.
                optim_g.zero_grad(set_to_none=True)
                last_layer = _get_last_layer(model)
                g_loss_tensor, g_log = loss_module(
                    out.vq_loss, x, out.recon,
                    optimizer_idx=0, global_step=step,
                    last_layer=last_layer, split="train",
                )
                g_loss_tensor.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim_g.step()

                # Discriminator branch — re-run forward to detach properly.
                if optim_d is not None:
                    optim_d.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        out_d = model(x)
                    d_loss_tensor, d_log = loss_module(
                        out_d.vq_loss, x, out_d.recon,
                        optimizer_idx=1, global_step=step, last_layer=None, split="train",
                    )
                    d_loss_tensor.backward()
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            loss_module.discriminator.parameters(), grad_clip,
                        )
                    optim_d.step()
                else:
                    d_log = {}

                if step % log_every == 0:
                    fmt_log = {**{k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)
                                    for k, v in g_log.items()},
                               **{k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)
                                    for k, v in d_log.items()}}
                    logger.info(
                        "[vq_font/vqgan] step=%d g_total=%.4f nll=%.4f q=%.4f d_loss=%.4f",
                        step,
                        fmt_log.get("train/total_loss", 0.0),
                        fmt_log.get("train/nll_loss", 0.0),
                        fmt_log.get("train/quant_loss", 0.0),
                        fmt_log.get("train/disc_loss", 0.0),
                    )

            step += 1
            if step >= max_steps or args.dry_run:
                break
        if step >= max_steps or args.dry_run:
            break

    if ckpt_dir is not None and not args.dry_run:
        path = Path(ckpt_dir) / "vqgan_last.pt"
        _save_state_with_cfg(
            state=model.state_dict(),
            cfg_dict=asdict(vqgan_cfg),
            ckpt_path=path,
        )
        logger.info("[vq_font/vqgan] saved checkpoint -> %s", path)
    logger.info("[vq_font/vqgan] done; final_step=%d dry_run=%s", step, args.dry_run)
    return 0


def _build_optimizer(
    model: VQFont,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
) -> torch.optim.Optimizer:
    """Adam over the trainable subset (transformer + partial VQGAN decoder).

    Stage 1+ uses Adam(0.0, 0.9) per ``cfgs/custom.yaml:24`` — very low β₁
    is the standard "GAN-style" / token-prediction recipe.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=lr, betas=betas, weight_decay=weight_decay)


def _save_transformer_ckpt(
    *,
    model: VQFont,
    vqgan_cfg: VQGANConfig,
    tr_cfg: TransformerConfig,
    ckpt_dir: Path,
) -> Path:
    """Save transformer + frozen VQGAN weights with JSON sidecar configs."""
    path = ckpt_dir / "transformer_last.pt"
    state = {
        "transformer": model.transformer.state_dict(),
        "vqgan": model.vqgan.state_dict(),
    }
    cfg_dict = {
        "vqgan": asdict(vqgan_cfg),
        "transformer": asdict(tr_cfg),
    }
    _save_state_with_cfg(state=state, cfg_dict=cfg_dict, ckpt_path=path)
    return path


def _try_load_vqgan_warmstart(
    model: VQFont, *, train_cfg: dict[str, Any]
) -> None:
    """Best-effort load of a Stage-0 VQGAN checkpoint for warm-start."""
    vqgan_ckpt = train_cfg.get("vqgan_ckpt")
    if not vqgan_ckpt:
        return
    ckpt_path = resolve_path(vqgan_ckpt, base=Path(__file__).resolve().parents[3])
    if ckpt_path.exists():
        _load_vqgan_ckpt(model, ckpt_path, strict=False)
        logger.info("[vq_font/transformer] loaded VQGAN ckpt: %s", ckpt_path)
    else:
        logger.warning(
            "[vq_font/transformer] VQGAN ckpt not found at %s — using random init",
            ckpt_path,
        )


def _run_transformer_stage(
    args: argparse.Namespace,
    *,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    paths: BackendPaths,
) -> int:
    device = torch.device(args.device)
    _seed_everything(int(train_cfg.get("seed", 42)))
    vqgan_cfg = _vqgan_cfg_from_yaml(model_cfg)
    tr_cfg = _transformer_cfg_from_yaml(model_cfg, vqgan_cfg)
    cfg = VQFontConfig(vqgan=vqgan_cfg, transformer=tr_cfg)
    # Partial freeze by default (encoder + late decoder + codebook frozen;
    # early decoder + post_quant trainable). See model.VQFont.
    freeze_mode = str(train_cfg.get("freeze_vqgan", "partial"))
    model = build_vq_font(cfg, freeze_vqgan=freeze_mode).to(device)
    _try_load_vqgan_warmstart(model, train_cfg=train_cfg)

    loader = _build_dataloader(
        args=args, data_cfg=data_cfg, model_cfg=cfg, train_cfg=train_cfg, paths=paths
    )
    lr = float(train_cfg.get("g_lr", train_cfg.get("learning_rate", 2.0e-4)))
    betas = _parse_adam_betas(train_cfg, default=(0.0, 0.9))
    optim = _build_optimizer(
        model,
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        betas=betas,
    )

    # Optional StepLR (matches ``cfgs/custom.yaml: step_size=10000, gamma=0.95``).
    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = None
    if sched_cfg.get("type", "step") == "step" and int(sched_cfg.get("step_size", 0)) > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim,
            step_size=int(sched_cfg.get("step_size", 10000)),
            gamma=float(sched_cfg.get("gamma", 0.95)),
        )

    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    max_steps = int(train_cfg.get("max_steps", 1 if args.dry_run else 1_500_000))
    log_every = int(train_cfg.get("log_every", 100))
    ckpt_dir = train_cfg.get("ckpt_dir")
    if ckpt_dir is not None:
        ckpt_dir = resolve_path(ckpt_dir, base=Path(__file__).resolve().parents[3])
        os.makedirs(ckpt_dir, exist_ok=True)

    logger.info(
        "[vq_font/transformer] device=%s bs=%s lr=%s betas=%s steps=%d K=%d latent=%d refs=%d freeze=%s",
        device, train_cfg.get("batch_size"), lr, betas, max_steps,
        vqgan_cfg.num_embeddings, tr_cfg.latent_resolution, tr_cfg.num_refs, freeze_mode,
    )

    model.train()
    if freeze_mode != "none":
        model.vqgan.eval()  # InstanceNorm doesn't track stats; this is mainly a no-op
    step = 0
    for _epoch in range(int(train_cfg.get("max_epochs", 1))):
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            optim.zero_grad(set_to_none=True)
            loss, log = transformer_compute_loss(model=model, batch=batch)
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
            optim.step()
            if scheduler is not None:
                scheduler.step()
            if step % log_every == 0:
                logger.info(
                    "[vq_font/transformer] step=%d loss=%.4f token_ce=%.4f top1=%.2f%%",
                    step, log["loss_total"], log["loss_token"], log["token_acc"] * 100,
                )
            step += 1
            if step >= max_steps or args.dry_run:
                break
        if step >= max_steps or args.dry_run:
            break

    if ckpt_dir is not None and not args.dry_run:
        path = _save_transformer_ckpt(
            model=model, vqgan_cfg=vqgan_cfg, tr_cfg=tr_cfg, ckpt_dir=Path(ckpt_dir),
        )
        logger.info("[vq_font/transformer] saved checkpoint -> %s", path)
    logger.info(
        "[vq_font/transformer] done; final_step=%d dry_run=%s", step, args.dry_run,
    )
    return 0


def main(
    args: argparse.Namespace,
    *,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    paths: BackendPaths,
) -> int:
    """Dispatch on ``train_cfg['stage']``."""
    stage = str(train_cfg.get("stage", "transformer")).lower()
    if stage in {"vqgan", "stage_0", "0"}:
        return _run_vqgan_stage(
            args, data_cfg=data_cfg, model_cfg=model_cfg, train_cfg=train_cfg, paths=paths
        )
    if stage in {"transformer", "stage_1", "1", "stage_a", "a"}:
        return _run_transformer_stage(
            args, data_cfg=data_cfg, model_cfg=model_cfg, train_cfg=train_cfg, paths=paths
        )
    raise ValueError(f"Unknown stage: {stage!r}; expected 'vqgan' or 'transformer'")
