"""FontDiffuser training — blind reimplementation.

Plumbed for the shared entrypoint ``paper_reimpl_shared.runner.entrypoint``:
  ``main(args, *, data_cfg, model_cfg, train_cfg, paths)``.

Loss = L_simple (DDPM denoising) + λ_scr * L_scr (style contrastive).
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
from paper_reimpl_shared.data.legacy import (
    collate_calligraphy_batch,
)
from paper_reimpl_shared.data.manifest import BackendPaths
from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion

from .dataset import build_dataset
from .model import (
    FontDiffuser,
    FontDiffuserConfig,
    StyleExtractor,
    build_fontdiffuser,
)

__all__ = ["compute_loss", "main"]


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------


def _style_contrastive_loss(
    z_pred: torch.Tensor,
    z_true: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss (NT-Xent variant).

    Treats z_true as the anchors and z_pred as the queries. For each query i,
    the positive set is {j : labels[j] == labels[i]} in z_true. All other
    z_true entries are negatives. Implements ``log(sum exp(sim/τ))`` partition
    over the anchor pool.

    This is the paper's SCR objective in spirit (paper §3 "SCR contrastive
    style loss with same-char-diff-style as negatives") — we substitute
    writer/style_family id for "diff-style" since "same-char-diff-style" is
    not constructible inside an iid batch without batch sampling logic.
    """
    sim = z_pred @ z_true.t() / temperature  # [B, B]
    label_eq = labels.unsqueeze(1) == labels.unsqueeze(0)  # [B, B]
    n_pos = label_eq.float().sum(dim=1).clamp_min(1.0)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_log_prob = (log_prob * label_eq.float()).sum(dim=1) / n_pos
    return -pos_log_prob.mean()


def compute_loss(
    *,
    model: FontDiffuser,
    diffusion: GaussianDiffusion,
    batch: dict[str, torch.Tensor],
    scr_extractor: StyleExtractor | None,
    scr_weight: float,
    cfg_drop_prob: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute L_simple + λ_scr * L_scr.

    Args:
        model: FontDiffuser.
        diffusion: shared GaussianDiffusion (prediction_target='epsilon'
            recommended for FontDiffuser; ``x0`` also supported).
        batch: dict with keys ``image``, ``content``, ``refs`` (or
            ``ref_images``), and a style-label tensor (``writer_id`` /
            ``style_family_id``).
        scr_extractor: frozen module producing L2-normalized embeddings.
            If ``None`` or ``scr_weight == 0``, the SCR term is skipped.
        scr_weight: λ_scr.
        cfg_drop_prob: probability of dropping the style reference (for
            classifier-free guidance training).
    """
    x0 = batch["image"]
    content = batch["content"]
    ref_images = batch.get("refs") if "refs" in batch else batch.get("ref_images")
    ref_valid = batch.get("ref_valid")
    if ref_valid is None and ref_images is not None and ref_images.numel() > 0:
        ref_valid = torch.ones(ref_images.shape[0], ref_images.shape[1], dtype=torch.bool, device=x0.device)

    if cfg_drop_prob > 0.0 and ref_valid is not None:
        # Per-sample uncond drop (FontDiffuser only conditions on content+ref;
        # we drop the ref only — content is required for source identity).
        drop = (torch.rand(ref_valid.shape[0], device=x0.device) < cfg_drop_prob)
        ref_valid = ref_valid.clone()
        ref_valid[drop, :] = False

    diff_batch = diffusion.sample_training_batch(x0)
    model_pred = model(
        diff_batch.x_t,
        diff_batch.timesteps,
        content=content,
        ref_images=ref_images,
        ref_valid=ref_valid,
    )
    loss_simple = F.mse_loss(model_pred, diff_batch.target, reduction="mean")

    log = {"loss_total": 0.0, "loss_simple": float(loss_simple.detach().cpu()), "loss_scr": 0.0}

    loss_scr = torch.zeros((), device=x0.device, dtype=x0.dtype)
    if scr_extractor is not None and scr_weight > 0.0:
        # Reconstruct predicted x0 from epsilon (or use pred directly if x0-target).
        x0_pred = diffusion.predict_x0(diff_batch.x_t, diff_batch.timesteps, model_pred)
        # Style labels: prefer writer_id, fall back to style_family_id.
        labels = batch.get("writer_id")
        if labels is None:
            labels = batch.get("style_family_id")
        if labels is None:
            labels = torch.zeros(x0.shape[0], dtype=torch.long, device=x0.device)
        with torch.no_grad():
            z_true = scr_extractor(x0)
        z_pred = scr_extractor(x0_pred)
        loss_scr = _style_contrastive_loss(z_pred, z_true, labels)
        log["loss_scr"] = float(loss_scr.detach().cpu())

    loss_total = loss_simple + scr_weight * loss_scr
    log["loss_total"] = float(loss_total.detach().cpu())
    return loss_total, log


# --------------------------------------------------------------------------------------
# Trainer wrapper
# --------------------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_cfg_from_yaml(model_cfg: dict[str, Any]) -> FontDiffuserConfig:
    m = model_cfg.get("model", model_cfg)
    return FontDiffuserConfig(
        image_size=int(m.get("image_size", 128)),
        in_channels=int(m.get("in_channels", 1)),
        content_channels=int(m.get("content_channels", 1)),
        ref_channels=int(m.get("ref_channels", 1)),
        base_channels=int(m.get("base_channels", 64)),
        channel_mult=tuple(int(x) for x in m.get("channel_mult", [1, 2, 4, 4])),
        attn_resolutions=tuple(int(x) for x in m.get("attn_resolutions", [16])),
        num_res_blocks=int(m.get("num_res_blocks", 2)),
        time_embed_dim=int(m.get("time_embed_dim", 256)),
        style_embed_dim=int(m.get("style_embed_dim", 256)),
        num_heads=int(m.get("num_heads", 4)),
        dropout=float(m.get("dropout", 0.0)),
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


def _build_dataloader(
    *,
    args,
    data_cfg: dict[str, Any],
    model_cfg: FontDiffuserConfig,
    train_cfg: dict[str, Any],
    paths: BackendPaths,
) -> DataLoader:
    """Pick between synthetic, ttf-pretrain, or manifest-backed datasets.

    For Stage A (TTF pretrain) and the smoke entrypoint we fall back to the
    synthetic dataset; real data plumbing for Stages B/C consumes manifest
    JSONLs via the shared ``CalligraphyJsonlDataset``.
    """
    dataset = build_dataset(
        args=args,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        paths=paths,
    )
    bs = int(train_cfg.get("batch_size", 4))
    nw = int(train_cfg.get("num_workers", 0))
    if getattr(args, "dry_run", False):
        # Dry-run path keeps everything in-process — multiprocessing here
        # would pickle local closures and choke on synthetic datasets.
        nw = 0
    max_refs = int(data_cfg.get("max_refs", 1))
    collate = _CollateWithRefs(max_refs)
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=False,
        num_workers=nw,
        collate_fn=collate,
    )


class _CollateWithRefs:
    """Picklable wrapper around ``collate_calligraphy_batch``.

    Module-level class so multiprocessing dataloader workers can serialize it
    (closures defined inside ``_build_dataloader`` cannot be pickled).
    """

    def __init__(self, max_refs: int) -> None:
        self.max_refs = max_refs

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_calligraphy_batch(batch, max_refs=self.max_refs)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """Move tensors to device; non-tensors are passed through."""
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    # Shared collate emits ``ref_images``; compute_loss accepts both keys.
    if "ref_images" in out and "refs" not in out:
        out["refs"] = out["ref_images"]
    return out


def main(args, *, data_cfg, model_cfg, train_cfg, paths: BackendPaths) -> int:
    """Entrypoint dispatched from paper_reimpl_shared.runner.entrypoint."""
    device = torch.device(args.device)
    _seed_everything(int(train_cfg.get("seed", 42)))

    cfg = _model_cfg_from_yaml(model_cfg)
    model = build_fontdiffuser(cfg).to(device)
    diffusion = _build_diffusion(train_cfg, device=device)

    scr_weight = float(train_cfg.get("scr_weight", 0.0))
    scr_extractor = None
    if scr_weight > 0.0:
        scr_extractor = StyleExtractor(
            in_channels=cfg.in_channels,
            embed_dim=int(train_cfg.get("scr_embed_dim", 128)),
        ).to(device)
        scr_extractor.eval()
        for p in scr_extractor.parameters():
            p.requires_grad = False

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
        f"[fontdiffuser] device={device} steps={max_steps} bs={train_cfg.get('batch_size')} "
        f"lr={lr} scr_weight={scr_weight} cfg_drop={cfg_drop} timesteps={diffusion.timesteps} "
        f"pred={diffusion.prediction_target} schedule={diffusion.beta_schedule}"
    )

    model.train()
    step = 0
    for epoch in range(int(train_cfg.get("max_epochs", 1))):
        for batch in loader:
            batch = _move_batch(batch, device)
            optim.zero_grad(set_to_none=True)
            loss, log = compute_loss(
                model=model,
                diffusion=diffusion,
                batch=batch,
                scr_extractor=scr_extractor,
                scr_weight=scr_weight,
                cfg_drop_prob=cfg_drop,
            )
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            if step % log_every == 0:
                print(
                    f"[fontdiffuser] step={step} loss_total={log['loss_total']:.4f} "
                    f"loss_simple={log['loss_simple']:.4f} loss_scr={log['loss_scr']:.4f}"
                )
            step += 1
            if step >= max_steps or args.dry_run:
                break
        if step >= max_steps or args.dry_run:
            break

    if ckpt_dir is not None and not args.dry_run:
        path = Path(ckpt_dir) / "fontdiffuser_last.pt"
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, path)
        print(f"[fontdiffuser] saved checkpoint -> {path}")

    print(f"[fontdiffuser] done; final_step={step} dry_run={args.dry_run}")
    return 0
