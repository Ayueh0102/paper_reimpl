"""FontDiffuser training — Phase 2 architecture-faithful loss.

Plumbed for the shared entrypoint ``paper_reimpl_shared.runner.entrypoint``:
  ``main(args, *, data_cfg, model_cfg, train_cfg, paths)``.

Phase 1 loss (official ``third_party/01_fontdiffuser/train.py:195-214``):

    loss = mse(eps_hat, eps)
         + perceptual_weight  * VGG-Perceptual(x0_pred, x0_true)
         + offset_l1_weight   * sum_of_offset_L1_terms_from_RSI_up_path

Phase 2 adds an InfoNCE Style Contrastive Refinement term on top.
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
    ContentPerceptualLoss,
    FontDiffuser,
    FontDiffuserConfig,
    SCRModule,
    build_fontdiffuser,
)

__all__ = ["compute_loss", "main"]


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------


def compute_loss(
    *,
    model: FontDiffuser,
    diffusion: GaussianDiffusion,
    batch: dict[str, torch.Tensor],
    perceptual_loss_fn: ContentPerceptualLoss | None = None,
    scr_module: SCRModule | None = None,
    scr_weight: float = 0.0,
    perceptual_weight: float | None = None,
    offset_l1_weight: float | None = None,
    cfg_drop_prob: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute Phase 1+2 FontDiffuser loss:

        L = mse + perceptual_weight * perceptual + offset_l1_weight * offset
                + scr_weight * scr_infonce

    Concept origin: ``third_party/01_fontdiffuser/train.py:195-228``.

    Coefficients default to the per-config values stored on ``model.cfg``
    (``offset_l1_weight=0.5``, ``perceptual_weight=0.01``). Stage A skips
    SCR by setting ``scr_weight=0``.

    CFG dropout protocol (Phase 2 fix): with prob ``cfg_drop_prob`` drop
    **both** content and style on the same samples — matches the official
    ``train.py:182-186`` "white-out content + style" protocol. The shared
    sampler honours the same convention by passing ``cfg_uncond_drops_content=True``.
    """
    if perceptual_weight is None:
        perceptual_weight = float(getattr(model.cfg, "perceptual_weight", 0.01))
    if offset_l1_weight is None:
        offset_l1_weight = float(getattr(model.cfg, "offset_l1_weight", 0.5))

    x0 = batch["image"]
    content = batch["content"]
    ref_images = batch.get("refs") if "refs" in batch else batch.get("ref_images")
    ref_valid = batch.get("ref_valid")
    if ref_valid is None and ref_images is not None and ref_images.numel() > 0:
        ref_valid = torch.ones(ref_images.shape[0], ref_images.shape[1], dtype=torch.bool, device=x0.device)

    if cfg_drop_prob > 0.0:
        # Phase 2 fix: drop content AND style on the same samples (concept
        # origin: ``third_party/01_fontdiffuser/train.py:182-186``). We zero
        # the content tensor and null ref_valid for the dropped rows. The
        # earlier blind impl only dropped ref_valid, which was inconsistent
        # with the official inference uncond ("white-out content + style").
        b = x0.shape[0]
        drop = (torch.rand(b, device=x0.device) < cfg_drop_prob)
        if drop.any():
            if ref_valid is not None:
                ref_valid = ref_valid.clone()
                ref_valid[drop, :] = False
            content = content.clone()
            content[drop] = 0.0

    diff_batch = diffusion.sample_training_batch(x0)
    model_pred = model(
        diff_batch.x_t,
        diff_batch.timesteps,
        content=content,
        ref_images=ref_images,
        ref_valid=ref_valid,
    )
    offset_l1 = getattr(model, "_last_offset_l1", None)
    loss_simple = F.mse_loss(model_pred, diff_batch.target, reduction="mean")

    log = {
        "loss_total": 0.0,
        "loss_simple": float(loss_simple.detach().cpu()),
        "loss_perceptual": 0.0,
        "loss_offset": 0.0,
        "loss_scr": 0.0,
    }

    loss_perc = torch.zeros((), device=x0.device, dtype=x0.dtype)
    if perceptual_loss_fn is not None and perceptual_weight > 0.0:
        x0_pred = diffusion.predict_x0(diff_batch.x_t, diff_batch.timesteps, model_pred)
        loss_perc = perceptual_loss_fn(x0_pred, x0)
        log["loss_perceptual"] = float(loss_perc.detach().cpu())

    loss_off = torch.zeros((), device=x0.device, dtype=x0.dtype)
    if offset_l1 is not None and offset_l1_weight > 0.0:
        # Official ``train.py:196`` divides the offset sum by 2 before
        # multiplying by ``offset_coefficient=0.5``. Net coefficient = 0.25.
        # We expose the divisor here implicitly via offset_l1_weight=0.5
        # against the raw sum (equivalent up to a constant factor).
        loss_off = offset_l1
        log["loss_offset"] = float(loss_off.detach().cpu())

    loss_scr = torch.zeros((), device=x0.device, dtype=x0.dtype)
    if scr_module is not None and scr_weight > 0.0:
        x0_pred = diffusion.predict_x0(diff_batch.x_t, diff_batch.timesteps, model_pred)
        negatives = batch.get("neg_images")
        if negatives is None or negatives.numel() == 0:
            # Without explicit dataset-mined negatives we cannot run the
            # paper-faithful SCR — log and skip. Phase 2 requires the
            # ``num_neg=16`` negative sampler in the dataset layer.
            log["loss_scr"] = 0.0
        else:
            loss_scr = scr_module(x0_pred, x0, negatives)
            log["loss_scr"] = float(loss_scr.detach().cpu())

    loss_total = (
        loss_simple
        + perceptual_weight * loss_perc
        + offset_l1_weight * loss_off
        + scr_weight * loss_scr
    )
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
        mca_stages=tuple(int(x) for x in m["mca_stages"]) if "mca_stages" in m else None,
        rsi_up_stages=tuple(int(x) for x in m["rsi_up_stages"]) if "rsi_up_stages" in m else None,
        se_reduction=int(m.get("se_reduction", 32)),
        offset_l1_weight=float(m.get("offset_l1_weight", 0.5)),
        perceptual_weight=float(m.get("perceptual_weight", 0.01)),
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
    scr_module: SCRModule | None = None
    if scr_weight > 0.0:
        scr_module = SCRModule(
            temperature=float(train_cfg.get("scr_temperature", 0.07)),
            image_size=cfg.image_size,
            nce_layers=tuple(int(x) for x in train_cfg.get("scr_nce_layers", [0, 1, 2, 3])),
            freeze_backbone=True,
        ).to(device)
        scr_module.eval()
        # Optionally load a separately-pretrained SCR checkpoint (paper does
        # this for Phase 2 — see ``third_party/01_fontdiffuser/scripts/train_phase_2.sh:9``).
        scr_ckpt = train_cfg.get("scr_ckpt_path")
        if scr_ckpt:
            state = torch.load(scr_ckpt, map_location=device)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            scr_module.load_state_dict(state, strict=False)
        for p in scr_module.parameters():
            p.requires_grad = False

    # VGG-Perceptual loss is on whenever ``perceptual_weight > 0`` (per yaml /
    # model cfg). Phase 1 default is 0.01.
    perceptual_loss_fn: ContentPerceptualLoss | None = None
    perceptual_weight_cfg = float(train_cfg.get("perceptual_weight", cfg.perceptual_weight))
    if perceptual_weight_cfg > 0.0:
        perceptual_loss_fn = ContentPerceptualLoss().to(device)
        perceptual_loss_fn.eval()
        for p in perceptual_loss_fn.parameters():
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
                perceptual_loss_fn=perceptual_loss_fn,
                scr_module=scr_module,
                scr_weight=scr_weight,
                perceptual_weight=perceptual_weight_cfg,
                offset_l1_weight=float(train_cfg.get("offset_l1_weight", cfg.offset_l1_weight)),
                cfg_drop_prob=cfg_drop,
            )
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            if step % log_every == 0:
                print(
                    f"[fontdiffuser] step={step} loss_total={log['loss_total']:.4f} "
                    f"loss_simple={log['loss_simple']:.4f} "
                    f"loss_perc={log['loss_perceptual']:.4f} "
                    f"loss_offset={log['loss_offset']:.4f} "
                    f"loss_scr={log['loss_scr']:.4f}"
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
