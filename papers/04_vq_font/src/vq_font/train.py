"""VQ-Font training — blind reimplementation.

Two stages are dispatched by ``train_cfg.stage``:

* ``stage: vqgan`` (Stage 0, paper-cited 200k iters)
    Trains the VQGAN font codebook end-to-end with reconstruction + VQ loss::

        L_vqgan = L_recon (L1) + L_perc (optional) + λ_vq * L_vq

    Phase 1 reimpl uses pure L1 reconstruction + the codebook commitment
    losses surfaced by ``VectorQuantize`` — no perceptual / GAN loss, which
    keeps the smoke test fast and the implementation surface small. Hooks
    are left in for adding them later (see ``reports/blind_impl.md``).

* ``stage: transformer`` (Stage 1+, paper-cited 300k iters)
    Trains the Token Prior Refinement Transformer with cross-entropy on the
    target codebook indices plus an auxiliary SSEM structure-classification
    loss::

        L_total = L_token_ce + λ_struct * L_struct_ce

    The VQGAN is loaded from ``train_cfg.vqgan_ckpt`` (Stage 0 checkpoint)
    and frozen.
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

__all__ = [
    "vqgan_compute_loss",
    "transformer_compute_loss",
    "main",
]


# --------------------------------------------------------------------------------------
# Config builders
# --------------------------------------------------------------------------------------


def _vqgan_cfg_from_yaml(model_cfg: dict[str, Any]) -> VQGANConfig:
    """Read either model_cfg['vqgan'] or model_cfg root keys."""
    raw = model_cfg.get("vqgan", model_cfg)
    return VQGANConfig(
        image_size=int(raw.get("image_size", 128)),
        in_channels=int(raw.get("in_channels", 1)),
        base_channels=int(raw.get("base_channels", 64)),
        channel_mult=tuple(int(x) for x in raw.get("channel_mult", [1, 2, 4])),
        z_channels=int(raw.get("z_channels", 256)),
        embed_dim=int(raw.get("embed_dim", 256)),
        num_embeddings=int(raw.get("num_embeddings", 1024)),
        commitment_weight=float(raw.get("commitment_weight", 0.25)),
        num_res_blocks=int(raw.get("num_res_blocks", 2)),
        dropout=float(raw.get("dropout", 0.0)),
    )


def _transformer_cfg_from_yaml(
    model_cfg: dict[str, Any], vqgan_cfg: VQGANConfig
) -> TransformerConfig:
    raw = model_cfg.get("transformer", {})
    # Default the codebook size + embed dim to match the VQGAN if not stated.
    return TransformerConfig(
        image_size=vqgan_cfg.image_size,
        latent_resolution=int(raw.get("latent_resolution", vqgan_cfg.out_resolution())),
        embed_dim=int(raw.get("embed_dim", vqgan_cfg.embed_dim)),
        num_blocks=int(raw.get("num_blocks", 6)),
        num_heads=int(raw.get("num_heads", 8)),
        mlp_ratio=float(raw.get("mlp_ratio", 4.0)),
        dropout=float(raw.get("dropout", 0.0)),
        num_refs=int(raw.get("num_refs", 3)),
        codebook_size=int(raw.get("codebook_size", vqgan_cfg.num_embeddings)),
        num_structures=int(raw.get("num_structures", 14)),
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
    """L_recon (L1 on pixels) + λ_vq * L_vq (commitment + codebook update).

    No GAN / perceptual term — see decision log entry on adversarial loss.
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
    structure_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L_token_ce + λ_struct * L_struct_ce.

    Token CE supervises the Transformer's output codebook-index logits
    against the **target** image's VQGAN-encoded indices (oracle quantization
    of the target glyph). The auxiliary structure CE supervises the SSEM
    head against the structure-id label.
    """
    target = batch["image"]
    # Phase 1 stand-in: use the target's content channel (source glyph render)
    # as the initial synthesis. Real pipeline would call an FFG module here.
    initial = batch.get("content")
    if initial is None or initial.numel() == 0:
        initial = target
    # If `content` has multi-channel cached fields but VQGAN expects single
    # channel, take the first channel (`bitmap` by convention).
    if initial.shape[1] != target.shape[1]:
        initial = initial[:, : target.shape[1]]
    refs = batch["ref_images"]
    structure_id = batch.get("structure_id")
    if structure_id is None:
        structure_id = torch.zeros(target.shape[0], dtype=torch.long, device=target.device)
    ref_valid = batch.get("ref_valid")

    token_logits, structure_logits = model.predict_token_logits(
        initial, refs, structure_id, ref_valid=ref_valid
    )

    with torch.no_grad():
        target_indices = model.encode_target_indices(target)  # [B, H_lat, W_lat]
    target_flat = target_indices.reshape(target.shape[0], -1)
    # token_logits: [B, N, K] -> reshape for CE.
    b, n, k = token_logits.shape
    loss_token = F.cross_entropy(
        token_logits.reshape(b * n, k),
        target_flat.reshape(b * n),
        reduction="mean",
    )
    loss_struct = F.cross_entropy(structure_logits, structure_id, reduction="mean")
    total = loss_token + structure_weight * loss_struct
    # Top-1 accuracy logged for sanity.
    with torch.no_grad():
        pred = token_logits.argmax(dim=-1)
        token_acc = (pred == target_flat).float().mean().item()
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_token": float(loss_token.detach().cpu()),
        "loss_struct": float(loss_struct.detach().cpu()),
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
    args,
    data_cfg: dict[str, Any],
    model_cfg,
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


def _load_vqgan_ckpt(
    model: VQFont | VQGAN, ckpt_path: str | Path, *, strict: bool = True
) -> None:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    if isinstance(model, VQFont):
        model.vqgan.load_state_dict(state, strict=strict)
    else:
        model.load_state_dict(state, strict=strict)


def _run_vqgan_stage(args, *, data_cfg, model_cfg, train_cfg, paths) -> int:
    device = torch.device(args.device)
    _seed_everything(int(train_cfg.get("seed", 42)))
    vqgan_cfg = _vqgan_cfg_from_yaml(model_cfg)
    model = build_vqgan(vqgan_cfg).to(device)

    loader = _build_dataloader(
        args=args, data_cfg=data_cfg, model_cfg=vqgan_cfg, train_cfg=train_cfg, paths=paths
    )
    lr = float(train_cfg.get("learning_rate", 4.0e-5))
    optim = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.999),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    recon_w = float(train_cfg.get("recon_weight", 1.0))
    vq_w = float(train_cfg.get("vq_weight", 1.0))
    max_steps = int(train_cfg.get("max_steps", 1 if args.dry_run else 200_000))
    log_every = int(train_cfg.get("log_every", 100))
    ckpt_dir = train_cfg.get("ckpt_dir")
    if ckpt_dir is not None:
        ckpt_dir = resolve_path(ckpt_dir, base=Path(__file__).resolve().parents[3])
        os.makedirs(ckpt_dir, exist_ok=True)

    print(
        f"[vq_font/vqgan] device={device} bs={train_cfg.get('batch_size')} lr={lr} "
        f"steps={max_steps} K={vqgan_cfg.num_embeddings} z_grid={vqgan_cfg.out_resolution()} "
        f"recon_w={recon_w} vq_w={vq_w}"
    )

    model.train()
    step = 0
    for _epoch in range(int(train_cfg.get("max_epochs", 1))):
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            optim.zero_grad(set_to_none=True)
            loss, log = vqgan_compute_loss(
                model=model, batch=batch, recon_weight=recon_w, vq_weight=vq_w
            )
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            if step % log_every == 0:
                print(
                    f"[vq_font/vqgan] step={step} loss={log['loss_total']:.4f} "
                    f"recon={log['loss_recon']:.4f} vq={log['loss_vq']:.4f}"
                )
            step += 1
            if step >= max_steps or args.dry_run:
                break
        if step >= max_steps or args.dry_run:
            break

    if ckpt_dir is not None and not args.dry_run:
        path = Path(ckpt_dir) / "vqgan_last.pt"
        torch.save({"model": model.state_dict(), "cfg": vqgan_cfg.__dict__}, path)
        print(f"[vq_font/vqgan] saved checkpoint -> {path}")
    print(f"[vq_font/vqgan] done; final_step={step} dry_run={args.dry_run}")
    return 0


def _run_transformer_stage(args, *, data_cfg, model_cfg, train_cfg, paths) -> int:
    device = torch.device(args.device)
    _seed_everything(int(train_cfg.get("seed", 42)))
    vqgan_cfg = _vqgan_cfg_from_yaml(model_cfg)
    tr_cfg = _transformer_cfg_from_yaml(model_cfg, vqgan_cfg)
    cfg = VQFontConfig(vqgan=vqgan_cfg, transformer=tr_cfg)
    model = build_vq_font(cfg, freeze_vqgan=True).to(device)

    vqgan_ckpt = train_cfg.get("vqgan_ckpt")
    if vqgan_ckpt:
        ckpt_path = resolve_path(vqgan_ckpt, base=Path(__file__).resolve().parents[3])
        if ckpt_path.exists():
            _load_vqgan_ckpt(model, ckpt_path, strict=False)
            print(f"[vq_font/transformer] loaded VQGAN ckpt: {ckpt_path}")
        else:
            print(f"[vq_font/transformer] WARN: VQGAN ckpt not found at {ckpt_path} — using random init")

    loader = _build_dataloader(
        args=args, data_cfg=data_cfg, model_cfg=cfg, train_cfg=train_cfg, paths=paths
    )
    lr = float(train_cfg.get("learning_rate", 2.0e-4))
    # Only optimize the Transformer (vqgan frozen).
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.999),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    structure_w = float(train_cfg.get("structure_weight", 0.1))
    max_steps = int(train_cfg.get("max_steps", 1 if args.dry_run else 300_000))
    log_every = int(train_cfg.get("log_every", 100))
    ckpt_dir = train_cfg.get("ckpt_dir")
    if ckpt_dir is not None:
        ckpt_dir = resolve_path(ckpt_dir, base=Path(__file__).resolve().parents[3])
        os.makedirs(ckpt_dir, exist_ok=True)

    print(
        f"[vq_font/transformer] device={device} bs={train_cfg.get('batch_size')} lr={lr} "
        f"steps={max_steps} K={vqgan_cfg.num_embeddings} latent={tr_cfg.latent_resolution} "
        f"refs={tr_cfg.num_refs} structure_w={structure_w}"
    )

    # VQGAN stays in eval mode (frozen BN/dropout).
    model.train()
    model.vqgan.eval()
    step = 0
    for _epoch in range(int(train_cfg.get("max_epochs", 1))):
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            optim.zero_grad(set_to_none=True)
            loss, log = transformer_compute_loss(
                model=model, batch=batch, structure_weight=structure_w
            )
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
            optim.step()
            if step % log_every == 0:
                print(
                    f"[vq_font/transformer] step={step} loss={log['loss_total']:.4f} "
                    f"token_ce={log['loss_token']:.4f} struct_ce={log['loss_struct']:.4f} "
                    f"top1={log['token_acc']*100:.2f}%"
                )
            step += 1
            if step >= max_steps or args.dry_run:
                break
        if step >= max_steps or args.dry_run:
            break

    if ckpt_dir is not None and not args.dry_run:
        path = Path(ckpt_dir) / "transformer_last.pt"
        torch.save(
            {
                "transformer": model.transformer.state_dict(),
                "vqgan": model.vqgan.state_dict(),
                "vqgan_cfg": vqgan_cfg.__dict__,
                "transformer_cfg": tr_cfg.__dict__,
            },
            path,
        )
        print(f"[vq_font/transformer] saved checkpoint -> {path}")
    print(f"[vq_font/transformer] done; final_step={step} dry_run={args.dry_run}")
    return 0


def main(args, *, data_cfg, model_cfg, train_cfg, paths: BackendPaths) -> int:
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
