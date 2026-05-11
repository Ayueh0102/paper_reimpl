"""HFH-Font training entry point.

Invoked by ``paper_reimpl_shared.runner.entrypoint`` via dynamic import of
``hfh_font.train.main``. Supports the ``--dry-run`` + ``--synthetic`` smoke
contract: build model, load one batch (synthetic by default), run one
forward + one optimizer step, and exit.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from paper_reimpl_shared.data.manifest import BackendPaths
from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion
from torch.utils.data import DataLoader

from .dataset import ManifestNotFoundError, build_collate, build_dataset
from .model import ModelConfig, build_model

__all__ = ["main", "set_seed"]

_logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(args_device: str) -> torch.device:
    if args_device.startswith("cuda") and not torch.cuda.is_available():
        _logger.warning(
            "requested device %s but CUDA unavailable; falling back to CPU",
            args_device,
        )
        return torch.device("cpu")
    return torch.device(args_device)


def _latest_ckpt(ckpt_dir: Path) -> Path | None:
    """Return the newest ``*.pt`` file in ``ckpt_dir``, or None if empty/absent."""
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _maybe_load_ckpt(
    model: torch.nn.Module,
    *,
    init_ckpt: str | None,
    resume: bool,
    ckpt_dir: Path,
    train_cfg: dict[str, Any],
) -> None:
    """Load weights from ``--init-ckpt`` / ``--resume`` / ``train_cfg['init_from']``.

    Resolution order (first hit wins):
      1. ``--init-ckpt PATH``   — explicit warm-start path from CLI.
      2. ``--resume``           — latest ``*.pt`` in ``ckpt_dir``.
      3. ``train_cfg.init_from``— path declared in the train YAML (Stage B/C
         use this to chain off Stage A).

    Loaded non-strict so partial state dicts (e.g. Stage A → Stage C with new
    modules) don't hard-fail. Missing/unexpected keys are logged.
    """
    path: Path | None = None
    source: str = ""
    if init_ckpt:
        path = Path(init_ckpt)
        source = "--init-ckpt"
    elif resume:
        path = _latest_ckpt(ckpt_dir)
        source = f"--resume (latest in {ckpt_dir})"
        if path is None:
            _logger.warning("--resume requested but no *.pt found in %s; starting from random init", ckpt_dir)
            return
    else:
        init_from = train_cfg.get("init_from")
        if init_from:
            path = Path(str(init_from))
            source = "train_cfg.init_from"

    if path is None:
        return
    if not path.exists():
        raise FileNotFoundError(f"[hfh_font] checkpoint not found: {path} (from {source})")

    state = torch.load(path, map_location="cpu")
    # Accept either a raw state_dict or a {"model": ..., "optimizer": ...} blob.
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        sd = state["model"]
    else:
        sd = state
    result = model.load_state_dict(sd, strict=False)
    _logger.info(
        "loaded init weights from %s (source=%s); missing=%d unexpected=%d",
        path,
        source,
        len(result.missing_keys),
        len(result.unexpected_keys),
    )
    if result.missing_keys:
        _logger.debug("missing keys: %s", result.missing_keys[:10])
    if result.unexpected_keys:
        _logger.debug("unexpected keys: %s", result.unexpected_keys[:10])


def _build_optimizer(model: torch.nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    betas = tuple(train_cfg.get("adam_betas", (0.9, 0.999)))
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )


def main(
    args: argparse.Namespace,
    *,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    paths: BackendPaths,
) -> int:
    """Phase-1 training entry."""
    # Configure root logging on first call so the entrypoint's stdout matches
    # the previous ``print()``-based contract. Safe to call repeatedly —
    # ``basicConfig`` is a no-op if a handler is already installed.
    logging.basicConfig(
        level=logging.INFO,
        format="[hfh_font] %(message)s",
    )
    seed = int(train_cfg.get("seed", 0))
    set_seed(seed)
    device = _resolve_device(args.device)
    _logger.info(
        "device=%s dry_run=%s synthetic=%s", device, args.dry_run, args.synthetic
    )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    cfg = ModelConfig.from_dict(model_cfg)
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    _logger.info("model built: %.2fM params, cfg=%s", n_params / 1e6, cfg)

    diffusion = GaussianDiffusion(
        timesteps=cfg.diffusion_timesteps,
        beta_schedule=str(train_cfg.get("beta_schedule", "linear")),
        beta_start=float(train_cfg.get("beta_start", 1e-4)),
        beta_end=float(train_cfg.get("beta_end", 2e-2)),
        prediction_target=cfg.diffusion_target,
        device=device,
    )

    # ------------------------------------------------------------------
    # Build dataset / dataloader
    # ------------------------------------------------------------------
    image_size = int(data_cfg.get("image_size", cfg.image_size))
    n_refs = int(data_cfg.get("n_refs", cfg.n_refs))
    content_channels = list(data_cfg.get("content_channels", ["bitmap", "sdf", "skeleton"]))
    try:
        dataset = build_dataset(
            data_cfg=data_cfg,
            backend=args.data_backend,
            synthetic=bool(args.synthetic),
            paths=paths,
            image_size=image_size,
            content_channels=content_channels,
            n_refs=n_refs,
        )
    except ManifestNotFoundError as exc:
        if args.dry_run or args.synthetic:
            _logger.warning(
                "manifest unavailable (%s); falling back to synthetic for dry-run", exc
            )
            data_cfg = {**data_cfg, "manifest": None}
            dataset = build_dataset(
                data_cfg=data_cfg,
                backend=args.data_backend,
                synthetic=True,
                paths=paths,
                image_size=image_size,
                content_channels=content_channels,
                n_refs=n_refs,
            )
        else:
            raise

    batch_size = int(train_cfg.get("batch_size", 8))
    if args.dry_run:
        batch_size = min(batch_size, 2)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not args.dry_run,
        num_workers=0,  # smoke / dry-run uses zero workers
        collate_fn=build_collate(n_refs=n_refs),
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Resolve checkpoint dir
    # ------------------------------------------------------------------
    ckpt_dir_raw = train_cfg.get("ckpt_dir", "outputs/hfh_font/default")
    ckpt_dir = Path(ckpt_dir_raw)
    if not ckpt_dir.is_absolute():
        # parents: [0]=hfh_font [1]=src [2]=02_hfh_font [3]=papers [4]=repo
        # root. yaml ckpt_dir is repo-root-relative, so use parents[4].
        repo_root = Path(__file__).resolve().parents[4]
        ckpt_dir = repo_root / ckpt_dir
    if not args.dry_run:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Warm-start / resume — must happen BEFORE optimizer is built so that
    # optimizer state (Adam moments) is created from the loaded params.
    # Resolution order: --init-ckpt > --resume > train_cfg.init_from.
    # ------------------------------------------------------------------
    _maybe_load_ckpt(
        model,
        init_ckpt=getattr(args, "init_ckpt", None),
        resume=bool(getattr(args, "resume", False)),
        ckpt_dir=ckpt_dir,
        train_cfg=train_cfg,
    )

    # ------------------------------------------------------------------
    # Build optimizer
    # ------------------------------------------------------------------
    optimizer = _build_optimizer(model, train_cfg)

    # ------------------------------------------------------------------
    # Train / dry-run loop
    # ------------------------------------------------------------------
    model.train()
    cfg_dropout = float(train_cfg.get("cfg_dropout", 0.1))
    stage = str(train_cfg.get("stage", "a")).lower()

    max_steps = 1 if args.dry_run else int(train_cfg.get("max_steps", 1))
    max_epochs = int(train_cfg.get("max_epochs", 1))
    log_every = int(train_cfg.get("log_every", 1))
    step = 0
    teacher: torch.nn.Module | None = None
    if stage == "c":
        # SDS distillation requires a frozen teacher with the same arch.
        teacher = build_model(cfg).to(device)
        teacher.load_state_dict(model.state_dict())
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    done = False
    for epoch in range(max_epochs):
        if done:
            break
        for batch in loader:
            batch = _to_device(batch, device)
            if stage == "c" and teacher is not None:
                losses = model.compute_sds_loss(batch, teacher, diffusion)  # type: ignore[arg-type]
            else:
                losses = model.compute_loss(batch, diffusion, cfg_dropout=cfg_dropout)
            loss = losses["loss"]
            if not torch.isfinite(loss):
                _logger.error("FATAL: non-finite loss at step %d: %s", step, loss.item())
                return 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip = float(train_cfg.get("grad_clip", 0.0))
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if step % log_every == 0:
                log_payload = {
                    k: float(v.detach()) for k, v in losses.items() if isinstance(v, torch.Tensor)
                }
                _logger.info("step=%d %s", step, log_payload)
            step += 1
            if step >= max_steps:
                done = True
                break

    if args.dry_run:
        _logger.info("dry-run OK — 1 step completed without errors.")
    else:
        ckpt_path = ckpt_dir / "hfh_font_last.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "cfg": cfg.__dict__,
            },
            ckpt_path,
        )
        _logger.info("saved checkpoint -> %s", ckpt_path)
    _logger.info("done; final_step=%d dry_run=%s", step, args.dry_run)
    return 0


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(
        "Run via paper_reimpl_shared.runner.entrypoint, not hfh_font.train directly."
    )
