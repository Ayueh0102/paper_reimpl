"""QT-Font training entry — Phase 2 paper-aligned.

Native training path
--------------------
1. Build a GT octree from the target glyph (3-class label map → sparse octree).
2. Sample ``t`` and corrupt the leaf-depth labels via D3PM uniform Gumbel-max.
3. Run :meth:`QTFontModel.predict_logits` with the GT topology and the noisy
   leaf labels, producing per-depth logits (2-way at inner depths, 3-way at
   the leaf).
4. Multi-depth CE against the GT split-mask + GT leaf labels.

Gradient accumulation
---------------------
The paper used ``accum=32`` to reach an effective batch of 1024. We expose
``grad_accum`` in the train YAML; ``accumulate_grad`` divides the loss before
.backward() so the .step() at the end of the cycle sees an averaged gradient.

Public surfaces
---------------
* :func:`compute_loss` — multi-depth CE + native diffusion noising.
* :func:`main`         — called by ``paper_reimpl_shared.runner.entrypoint``.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SyntheticConfig, build_dataset
from .losses import compute_multi_depth_ce
from .model import D3PMUniform, QTFontConfig, QTFontModel, build_qt_font
from .octree import OctreeBatch, build_octree_from_image

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Loss                                                                         #
# --------------------------------------------------------------------------- #


def compute_loss(
    *,
    model: QTFontModel,
    diffusion: D3PMUniform,
    batch: dict[str, torch.Tensor],
    # Legacy kwarg retained for shared harness — ignored in Phase 2.
    cfg_drop_prob: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-depth CE loss following ``losses/loss.py:axis_loss``.

    The batch dictionary is the standard ``make_synthetic_batch`` / manifest
    schema: ``image, content, refs, char_id, writer_id, script_id``. The
    categorical ids are accepted but **not used** by the paper-aligned model
    (paper conditioning_deltas C1).

    Parameters
    ----------
    cfg_drop_prob : float
        Retained for cross-paper signature compatibility; unused in Phase 2.
    """
    del cfg_drop_prob  # explicitly silence the back-compat kwarg

    cfg = model.cfg
    image = batch["image"]
    device = image.device
    B = image.shape[0]

    # 1) Build GT octree from the clean target glyph.
    gt_octree = build_octree_from_image(
        image, full_depth=cfg.full_depth, depth=cfg.depth
    ).to(device)

    # 2) Sample t and corrupt the leaf-depth labels.
    t = diffusion.sample_random_step(B, device=device)
    leaf = gt_octree.levels[cfg.depth]
    # Per-leaf timestep = the leaf's batch's t.
    t_per_leaf = t[leaf.batch_id]
    x0_label = leaf.leaf_label  # (N,)
    xt_label = diffusion.q_sample(x0_label, t_per_leaf)

    # 3) Build conditioning. Content + refs are optional and shape (B, C, H, W),
    #    (B, R, C, H, W). Both are interpreted as pixel tensors and re-binned to
    #    octrees.
    content_octree = None
    content = batch.get("content")
    if content is not None and cfg.use_content:
        if content.dim() == 4 and content.shape[1] != 1:
            content = content[:, :1]
        content_octree = build_octree_from_image(
            content, full_depth=cfg.full_depth, depth=cfg.depth
        ).to(device)
    style_octrees: list[OctreeBatch] | None = None
    # Note: do NOT use `or` here — `tensor or other` triggers the bool-coerce
    # error for multi-element tensors. Explicit None check instead.
    refs = batch.get("refs")
    if refs is None:
        refs = batch.get("ref_images")
    if refs is not None and cfg.use_style:
        R = refs.shape[1]
        style_octrees = [
            build_octree_from_image(
                refs[:, r, :1], full_depth=cfg.full_depth, depth=cfg.depth
            ).to(device)
            for r in range(R)
        ]
    cond = model.encode_conditioning(
        t.float(),
        content_octree=content_octree,
        style_octrees=style_octrees,
    )

    # 4) Forward & multi-depth CE.
    logits_per_depth = model.predict_logits(gt_octree, cond, noisy_leaf_label=xt_label)
    loss, log = compute_multi_depth_ce(logits_per_depth, gt_octree)
    log["mean_t"] = float(t.float().mean().item())
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
    channels = section.get("channels_per_depth")
    if channels is None:
        # Default mirrors the official 128/256 px width table.
        channels = (3, 512, 512, 256, 512, 256, 128, 64, 64, 64)
    return QTFontConfig(
        image_size=int(section.get("image_size", 128)),
        full_depth=int(section.get("full_depth", 4)),
        depth=int(section.get("depth", 7)),
        depth_stop=int(section.get("depth_stop", 4)),
        n_states=int(section.get("n_states", 3)),
        channels_per_depth=tuple(channels),
        cond_dim=int(section.get("cond_dim", 256)),
        timesteps=int(section.get("timesteps", 1000)),
        schedule=str(section.get("schedule", "cos")),
        use_style=bool(section.get("use_style", True)),
        use_content=bool(section.get("use_content", True)),
        # Legacy fields populated from data_cfg for back-compat smoke runs.
        in_channels=int(section.get("in_channels", 1)),
        content_channels=int(section.get("content_channels", 1)),
        ref_channels=int(section.get("ref_channels", 1)),
        char_vocab_size=int(data_cfg.get("char_vocab_size", 0)),
        writer_vocab_size=int(data_cfg.get("writer_vocab_size", 0)),
        script_vocab_size=int(data_cfg.get("script_vocab_size", 0)),
    )


def _build_synthetic_loader(
    qt_cfg: QTFontConfig, data_cfg: dict[str, Any], batch_size: int, num_workers: int
) -> DataLoader:
    syn_cfg = SyntheticConfig(
        length=int(data_cfg.get("synthetic_length", 32)),
        image_size=1 << qt_cfg.depth,
        in_channels=qt_cfg.in_channels,
        content_channels=qt_cfg.content_channels,
        n_refs=int(data_cfg.get("max_refs", 1)),
        char_vocab_size=max(1, qt_cfg.char_vocab_size),
        writer_vocab_size=max(1, qt_cfg.writer_vocab_size),
        script_vocab_size=max(1, qt_cfg.script_vocab_size),
        seed=int(data_cfg.get("seed", 0)),
    )
    ds = build_dataset(syn_cfg)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )


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
        schedule=str(diff_cfg.get("schedule", qt_cfg.schedule)),
        beta_start=float(diff_cfg.get("beta_start", 0.02)),
        beta_end=float(diff_cfg.get("beta_end", 1.0)),
    ).to(args.device)

    # Routing: synthetic (smoke / dry-run), ttf (real cross-font pretrain),
    # or manifest (real Stage B/C, not implemented).
    source = str(data_cfg.get("source", "synthetic")).lower()
    if args.synthetic or source == "synthetic":
        loader = _build_synthetic_loader(
            qt_cfg,
            data_cfg,
            batch_size=int(train_cfg.get("batch_size", 2)),
            num_workers=int(train_cfg.get("num_workers", 0)),
        )
    elif source == "ttf":
        from pathlib import Path as _P
        from .dataset import build_ttf_dataset

        fonts_root_cfg = data_cfg.get("fonts_root")
        if fonts_root_cfg:
            fonts_root = _P(str(fonts_root_cfg))
        else:
            fonts_root = paths.ttf_root.parent / "fonts_free"
        cache_cfg = data_cfg.get("supported_chars_cache")
        cache_path = _P(str(cache_cfg)) if cache_cfg else None
        ttf_ds = build_ttf_dataset(
            fonts_root=fonts_root,
            image_size=1 << qt_cfg.depth,
            content_channels=qt_cfg.content_channels,
            n_refs=int(data_cfg.get("max_refs", 1)),
            font_size_ratio=float(data_cfg.get("font_size_ratio", 0.85)),
            length=int(data_cfg.get("ttf_epoch_length", 10_000)),
            seed=int(train_cfg.get("seed", 42)),
            cjk_start=int(data_cfg.get("cjk_start", 0x4E00)),
            cjk_end=int(data_cfg.get("cjk_end", 0x9FFF)),
            char_cache_path=cache_path,
            font_ids=data_cfg.get("font_ids"),
            script_categories=data_cfg.get("script_categories"),
        )
        nw = int(train_cfg.get("num_workers", 0))
        loader = DataLoader(
            ttf_ds,
            batch_size=int(train_cfg.get("batch_size", 2)),
            shuffle=True,
            num_workers=nw,
            drop_last=False,
            persistent_workers=(nw > 0),
            pin_memory=True,
            prefetch_factor=4 if nw > 0 else None,
        )
    elif source == "manifest":
        from pathlib import Path as _P
        from .dataset import build_manifest_dataset

        manifest_name = data_cfg.get("manifest")
        if not manifest_name:
            raise ValueError(
                "data_cfg must contain `manifest: <file name>` when source=manifest"
            )
        manifest_path = paths.manifest_root / str(manifest_name)
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest missing: {manifest_path}")
        content_channels_list = list(data_cfg.get("content_channels", ["bitmap"]))
        manifest_ds = build_manifest_dataset(
            manifest_path=manifest_path,
            image_size=1 << qt_cfg.depth,
            content_channels=content_channels_list,
            max_refs=int(data_cfg.get("max_refs", 0)),
        )
        nw = int(train_cfg.get("num_workers", 0))
        loader = DataLoader(
            manifest_ds,
            batch_size=int(train_cfg.get("batch_size", 2)),
            shuffle=True,
            num_workers=nw,
            drop_last=False,
            persistent_workers=(nw > 0),
            pin_memory=True,
            prefetch_factor=4 if nw > 0 else None,
        )
    else:  # pragma: no cover - placeholder
        raise NotImplementedError(
            f"Unknown source={source}; supported: synthetic, ttf, manifest."
        )

    # Warm-start from --init-ckpt before optimizer is built so AdamW moments
    # are created from loaded params. Weights-only load with strict=False.
    init_ckpt = getattr(args, "init_ckpt", None)
    if init_ckpt:
        blob = torch.load(init_ckpt, map_location=args.device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        missing, unexpected = model.load_state_dict(state, strict=False)
        logging.info(
            "warm-start from %s (missing=%d unexpected=%d)",
            init_ckpt,
            len(missing),
            len(unexpected),
        )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        betas=(0.9, 0.999),
        # Paper uses wd=0.0 (configs/chinesefont_train.yaml:26).
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    logger.info("model params = %s", f"{sum(p.numel() for p in model.parameters()):,}")
    logger.info("qt_cfg = %s", asdict(qt_cfg))
    max_steps = int(train_cfg.get("max_steps", 10))
    max_epochs = int(train_cfg.get("max_epochs", 1))
    log_every = int(train_cfg.get("log_every", 1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    grad_accum = max(1, int(train_cfg.get("grad_accum", 1)))

    ckpt_dir_raw = train_cfg.get("ckpt_dir")
    ckpt_dir = None
    if ckpt_dir_raw is not None:
        from pathlib import Path as _P
        p = _P(str(ckpt_dir_raw))
        if not p.is_absolute():
            # parents: [0]=qt_font [1]=src [2]=05_qt_font [3]=papers [4]=repo
            p = _P(__file__).resolve().parents[4] / p
        if not args.dry_run:
            p.mkdir(parents=True, exist_ok=True)
        ckpt_dir = p

    step = 0
    micro = 0
    opt.zero_grad(set_to_none=True)
    done = False
    for epoch in range(max_epochs):
        if done:
            break
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss, log = compute_loss(model=model, diffusion=diffusion, batch=batch)
            # Mean-reduce across the accumulation cycle so the .step() gradient is
            # an unbiased estimate of the effective-batch gradient.
            (loss / grad_accum).backward()
            micro += 1
            if micro >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                micro = 0
                step += 1
                if step % log_every == 0:
                    logger.info("step=%d %s", step, log)
                if args.dry_run:
                    logger.info("dry-run: stop after 1 step")
                    return 0
                if step >= max_steps:
                    done = True
                    break

    if ckpt_dir is not None and not args.dry_run:
        ckpt_path = ckpt_dir / "qt_font_last.pt"
        torch.save(
            {"model": model.state_dict(), "step": step, "cfg": asdict(qt_cfg)},
            ckpt_path,
        )
        logger.info("saved checkpoint -> %s", ckpt_path)
    logger.info("training done; total_steps=%d", step)
    return 0
