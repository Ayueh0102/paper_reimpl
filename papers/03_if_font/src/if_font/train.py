"""IF-Font training — Phase 2 alignment.

Loss = `sq` (CE on next-VQ-token) + `sup_cl` (supervised contrastive over
MoCo style features). Phase-1's (CE, VQ commitment, recon MSE) triple is
gone — VQGAN is frozen pretrained, so commitment + recon do not apply.

Optimiser & schedule:
  * Two AdamW groups:
      - decoder ("netTransformer") params: betas=(0.9, 0.95), weight_decay=0.01
      - everything else (IDS embeddings, MoCo): betas=(0.9, 0.999)
  * OneCycleLR with `pct_start = 0.5 / max_epochs`, `final_div_factor = 10/25`.
  * No grad clip by default (official does not set one).
"""

from __future__ import annotations

import collections
import dataclasses
import os
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from paper_reimpl_shared.config import resolve_path
from paper_reimpl_shared.data.manifest import BackendPaths
from torch.utils.data import DataLoader

from . import losses
from .dataset import IFFontCollate, build_dataset, load_ids_lookup
from .ids import IDSResolver, IDSTokenizer
from .model import IFFont, IFFontConfig, VQTokenizerAdapter, VQTokenizerConfig, build_if_font

__all__ = ["MoCoCache", "compute_loss", "main"]


# --------------------------------------------------------------------------------------
# MoCo cache queue (replaces Phase-1 in-batch sup_cl)
# --------------------------------------------------------------------------------------


class MoCoCache:
    """Bounded FIFO cache of (cl_q+cl_m, font_id) pairs.

    Mirrors official `models/net2net_model.CacheManagerCL` (size=10 batches).
    """

    def __init__(self, max_batches: int = 10) -> None:
        self.max_batches = max_batches
        self.cl_buf: deque[torch.Tensor] = deque(maxlen=max_batches)
        self.id_buf: deque[torch.Tensor] = deque(maxlen=max_batches)

    def push(self, cl: torch.Tensor, font_id: torch.Tensor) -> None:
        self.cl_buf.append(cl.detach())
        self.id_buf.append(font_id.detach())

    def pop_concat(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.cl_buf:
            return None, None
        return torch.cat(list(self.cl_buf), dim=0), torch.cat(list(self.id_buf), dim=0)


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------


def compute_loss(
    *,
    model: IFFont,
    batch: dict[str, torch.Tensor],
    cache: MoCoCache | None = None,
    sup_cl_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """IF-Font Phase-2 loss: `sq + sup_cl_weight * sup_cl`.

    Official `Net2NetModel.training_step`:
        l_sq = losses.sq(logits, x)
        l_cl = losses.sup_cl(cl_s, labels=font_id) / 2     # weight 0.5
        return l_sq + l_cl

    The `/2` in the official is captured here via `sup_cl_weight=0.5`.

    `batch` must contain (Phase-2 collate emits all of these):
      * image, refs/ref_images, ids_token_ids, coverage_sim, font_id
    """
    image = batch["image"]
    ref_images = batch.get("refs") if "refs" in batch else batch.get("ref_images")
    ids_token_ids = batch["ids_token_ids"]
    coverage_sim = batch["coverage_sim"]
    font_id = batch.get("font_id", batch.get("writer_id"))

    if ref_images is None or ref_images.numel() == 0:
        raise ValueError("IF-Font Phase 2 requires at least one reference glyph.")

    out = model(
        target_image=image,
        ids_token_ids=ids_token_ids,
        ref_images=ref_images,
        coverage_sim=coverage_sim,
    )
    logits = out["logits"]
    target_ids = out["target_ids"]

    l_sq = losses.sq(logits, target_ids)

    l_cl_val = torch.zeros((), device=logits.device)
    if model.training and out["cl"] is not None and font_id is not None:
        cl = out["cl"]  # [B, 2, dim]
        if cache is not None:
            # Concat with stale entries first, then push the fresh batch.
            stale_cl, stale_id = cache.pop_concat()
            cache.push(cl, font_id)
            if stale_cl is not None and stale_id is not None:
                cl_all = torch.cat([cl, stale_cl.to(cl.device)], dim=0)
                id_all = torch.cat([font_id, stale_id.to(font_id.device)], dim=0)
            else:
                cl_all, id_all = cl, font_id
        else:
            cl_all, id_all = cl, font_id
        l_cl_val = losses.sup_cl(cl_all, labels=id_all)

    total = l_sq + sup_cl_weight * l_cl_val

    log = {
        "loss_total": float(total.detach().cpu()),
        "loss_sq": float(l_sq.detach().cpu()),
        "loss_cl": float(l_cl_val.detach().cpu()) if isinstance(l_cl_val, torch.Tensor) else 0.0,
    }
    return total, log


# --------------------------------------------------------------------------------------
# Trainer plumbing
# --------------------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _vq_cfg_from_yaml(m: dict[str, Any]) -> VQTokenizerConfig:
    vq = m.get("vq", {})
    return VQTokenizerConfig(
        image_size=int(m.get("image_size", 128)),
        in_channels=int(m.get("in_channels", 3)),
        embedding_dim=int(vq.get("embedding_dim", 4)),
        codebook_size=int(vq.get("codebook_size", 256)),
        downsample_factor=int(vq.get("downsample_factor", 8)),
    )


def _model_cfg_from_yaml(model_cfg: dict[str, Any]) -> IFFontConfig:
    m = model_cfg.get("model", model_cfg)
    vq_cfg = _vq_cfg_from_yaml(m)
    return IFFontConfig(
        image_size=int(m.get("image_size", 128)),
        in_channels=int(m.get("in_channels", 3)),
        vq=vq_cfg,
        ids_vocab_size=int(m.get("ids_vocab_size", 1024)),
        ids_max_len=int(m.get("ids_max_len", 35)),
        d_model=int(m.get("d_model", 384)),
        n_heads=int(m.get("n_heads", 8)),
        n_blocks=int(m.get("n_blocks", 10)),
        ffn_mult=int(m.get("ffn_mult", 4)),
        dropout=float(m.get("dropout", 0.1)),
        bias=bool(m.get("bias", False)),
        n_refs=int(m.get("n_refs", 3)),
    )


def _build_ids_resolver(train_cfg: dict[str, Any]) -> IDSResolver | None:
    """Try to load BabelStone + ids_iffont. Return None if files absent."""
    babel = train_cfg.get("babelstone_path") or "~/Char/datasets/ids/cn_mainland/babelstone_cjk_ids.txt"
    iff = train_cfg.get("ids_iffont_path") or "~/Char/datasets/ids/cn_mainland/ids_iffont.txt"
    bp = Path(babel).expanduser()
    ip = Path(iff).expanduser()
    if not bp.exists() and not ip.exists():
        return None
    return IDSResolver.load(level="radical", babelstone_path=bp, ids_iffont_path=ip)


def _warm_fit_tokenizer(
    tokenizer: IDSTokenizer,
    dataset,
    *,
    max_samples: int | None = None,
) -> None:
    n = len(dataset)
    if max_samples is not None:
        n = min(n, int(max_samples))
    ids_strings: list[str] = []
    for i in range(n):
        try:
            row = dataset[i]
        except Exception:  # pragma: no cover
            continue
        s = row.get("ids_string", "") if isinstance(row, dict) else ""
        if s:
            ids_strings.append(s)
    tokenizer.fit_from_strings(ids_strings)
    tokenizer.freeze()


def _build_dataloader(
    *,
    args,
    data_cfg: dict[str, Any],
    model_cfg: IFFontConfig,
    train_cfg: dict[str, Any],
    paths: BackendPaths,
    tokenizer: IDSTokenizer,
    ids_resolver: IDSResolver | None,
) -> DataLoader:
    ids_lookup = load_ids_lookup(data_cfg.get("ids_lookup_path"))
    dataset = build_dataset(
        args=args,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        paths=paths,
        ids_lookup=ids_lookup,
        ids_resolver=ids_resolver,
    )
    bs = int(train_cfg.get("batch_size", 4))
    nw = int(train_cfg.get("num_workers", 0))
    if getattr(args, "dry_run", False):
        nw = 0

    if not tokenizer.is_frozen:
        if ids_resolver is not None:
            tokenizer.fit_from_resolver(ids_resolver)
            tokenizer.freeze()
        else:
            _warm_fit_tokenizer(
                tokenizer,
                dataset,
                max_samples=int(train_cfg.get("tokenizer_warm_fit_max_samples", 0)) or None,
            )

    collate = IFFontCollate(
        tokenizer=tokenizer,
        max_refs=int(data_cfg.get("max_refs", model_cfg.n_refs)),
        ids_max_len=int(model_cfg.ids_max_len),
        in_channels=int(model_cfg.in_channels),
        fit_on_first_call=False,
    )
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=False,
        num_workers=nw,
        collate_fn=collate,
        persistent_workers=(nw > 0),
        pin_memory=True,
        prefetch_factor=4 if nw > 0 else None,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _configure_optimizers(model: IFFont, lr: float, weight_decay: float):
    """Two-group AdamW (matches official `Net2NetModel.configure_optimizers`)."""
    decoder_params = list(model.decoder.parameters())
    other_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("decoder.") and p.requires_grad
    ]
    optim = torch.optim.AdamW(
        [
            {"params": decoder_params, "betas": (0.9, 0.95), "weight_decay": weight_decay},
            {"params": other_params, "betas": (0.9, 0.999), "weight_decay": weight_decay},
        ],
        lr=lr,
    )
    return optim


def main(args, *, data_cfg, model_cfg, train_cfg, paths: BackendPaths) -> int:
    device = torch.device(args.device)
    _seed_everything(int(train_cfg.get("seed", 42)))

    cfg = _model_cfg_from_yaml(model_cfg)
    ids_resolver = _build_ids_resolver(train_cfg)

    tokenizer = IDSTokenizer.from_idc_only()
    if ids_resolver is not None:
        tokenizer.fit_from_resolver(ids_resolver)
    _IDS_VOCAB_HEADROOM = 4096
    cfg.ids_vocab_size = max(cfg.ids_vocab_size, tokenizer.vocab_size + _IDS_VOCAB_HEADROOM)

    # VQ tokenizer: priority order
    #   1. vqgan_local_path → load our own pretrained stub-adapter state_dict
    #      (saved by scripts/pretrain_vqgan.py).
    #   2. vqgan_path → load real CompVis vq-f8-n256 (needs taming-transformers).
    #   3. fallback → stub adapter with random init (Phase-2 collapses to sq=0).
    vq_path = train_cfg.get("vqgan_path")
    vq_local = train_cfg.get("vqgan_local_path")
    if vq_local:
        import torch as _torch
        vq_local_resolved = resolve_path(vq_local, base=Path(__file__).resolve().parents[4])
        vq_adapter = VQTokenizerAdapter(cfg.vq)
        blob = _torch.load(str(vq_local_resolved), map_location="cpu", weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        miss, unexp = vq_adapter.load_state_dict(state, strict=False)
        print(
            f"[if_font] loaded local pretrained VQGAN from {vq_local_resolved} "
            f"(missing={len(miss)} unexpected={len(unexp)})"
        )
        vq_adapter._freeze()
    elif vq_path:
        vq_adapter = VQTokenizerAdapter.from_pretrained_compvis(str(Path(vq_path).expanduser()))
    else:
        vq_adapter = VQTokenizerAdapter(cfg.vq)
    model = build_if_font(cfg, vq_adapter=vq_adapter).to(device)

    # Warm-start from --init-ckpt before optimizer build.
    # IMPORTANT: filter out vq_adapter.* keys when a fresh local VQGAN is
    # loaded — otherwise v1's random-stub VQGAN weights overwrite the new
    # pretrained tokenizer and we're back at codebook collapse.
    init_ckpt = getattr(args, "init_ckpt", None)
    if init_ckpt:
        import torch as _torch
        blob = _torch.load(init_ckpt, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        if vq_local:
            state = {k: v for k, v in state.items() if not k.startswith("vq.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[if_font] warm-start from {init_ckpt} "
            f"(missing={len(missing)} unexpected={len(unexpected)}; "
            f"vq_adapter.* filtered={bool(vq_local)})"
        )

    loader = _build_dataloader(
        args=args,
        data_cfg=data_cfg,
        model_cfg=cfg,
        train_cfg=train_cfg,
        paths=paths,
        tokenizer=tokenizer,
        ids_resolver=ids_resolver,
    )
    if tokenizer.vocab_size > cfg.ids_vocab_size:
        raise ValueError(
            f"IDS tokenizer vocab ({tokenizer.vocab_size}) exceeds model "
            f"ids_vocab_size ({cfg.ids_vocab_size})"
        )

    base_lr = float(train_cfg.get("base_learning_rate", 4.5e-6))
    bs = int(train_cfg.get("batch_size", 4))
    accum = int(train_cfg.get("accumulate_grad_batches", 1))
    # Official `run.py:23-25`: lr = accumulate * bs * base_lr.
    lr_raw = train_cfg.get("learning_rate", "")
    lr = float(lr_raw) if (lr_raw not in ("", None)) else base_lr * bs * accum
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    optim = _configure_optimizers(model, lr=lr, weight_decay=weight_decay)

    max_epochs = int(train_cfg.get("max_epochs", 15))
    max_steps_yaml = int(train_cfg.get("max_steps", 1 if args.dry_run else 1_000_000))
    steps_per_epoch = max(1, len(loader))
    total_steps = min(max_steps_yaml, max_epochs * steps_per_epoch) if not args.dry_run else 1

    pct_start = 0.5 / max(1, max_epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=lr,
        total_steps=max(2, total_steps),  # OneCycleLR needs >=2 steps
        pct_start=min(0.5, max(1e-3, pct_start)),
        final_div_factor=10.0 / 25.0,
    )

    sup_cl_weight = float(train_cfg.get("sup_cl_weight", 0.5))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    moco_cache_size = int(train_cfg.get("moco_cache_size", 10))
    cache = MoCoCache(max_batches=moco_cache_size)

    log_every = int(train_cfg.get("log_every", 50))

    ckpt_dir = train_cfg.get("ckpt_dir")
    if ckpt_dir is not None:
        ckpt_dir = resolve_path(ckpt_dir, base=Path(__file__).resolve().parents[4])
        os.makedirs(ckpt_dir, exist_ok=True)

    print(
        f"[if_font] device={device} max_steps={total_steps} bs={bs} lr={lr:.2e} "
        f"sup_cl_weight={sup_cl_weight} codebook={cfg.vq.codebook_size} "
        f"d_model={cfg.d_model} blocks={cfg.n_blocks} n_refs={cfg.n_refs} "
        f"vqgan_pretrained={bool(vq_local or vq_path)} local={bool(vq_local)}"
    )

    model.train()
    step = 0
    for epoch in range(max_epochs):
        for batch in loader:
            batch = _move_batch(batch, device)
            optim.zero_grad(set_to_none=True)
            loss, log = compute_loss(
                model=model,
                batch=batch,
                cache=cache,
                sup_cl_weight=sup_cl_weight,
            )
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            if step + 1 < scheduler.total_steps:
                scheduler.step()
            # Cosine momentum schedule for MoCo (official: epoch / max_epochs).
            model.moco_wrapper.momentum_update(epoch / max(1, max_epochs))

            if step % log_every == 0:
                print(
                    f"[if_font] epoch={epoch} step={step} total={log['loss_total']:.4f} "
                    f"sq={log['loss_sq']:.4f} cl={log['loss_cl']:.4f}"
                )
            step += 1
            if step >= total_steps or args.dry_run:
                break
        if step >= total_steps or args.dry_run:
            break

    if ckpt_dir is not None and not args.dry_run:
        path = Path(ckpt_dir) / "if_font_last.pt"
        torch.save(
            {"model": model.state_dict(), "cfg": dataclasses.asdict(cfg)},
            path,
        )
        print(f"[if_font] saved checkpoint -> {path}")

    print(f"[if_font] done; final_step={step} dry_run={args.dry_run}")
    return 0


# Silence unused-imports.
_ = collections
