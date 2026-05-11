"""IF-Font training — blind reimplementation.

Plumbed for the shared entrypoint
``paper_reimpl_shared.runner.entrypoint``:
  ``main(args, *, data_cfg, model_cfg, train_cfg, paths)``.

Loss (paper §3, p.1):
  L = ce_weight * L_AR + vq_weight * L_VQ + recon_weight * L_recon

  * L_AR    : cross-entropy on next-VQ-token prediction (the paper's main loss).
  * L_VQ    : VQ commitment loss (only active when VQ is being trained; ~ Stage A).
  * L_recon : MSE reconstruction loss from VQ decoder (active in Stage A).

Stage-A YAML sets ce_weight=0, vq_weight=1, recon_weight=1 → pure VQGAN
pretraining of the codebook.
Stage-B/C YAML sets ce_weight=1, vq_weight=0, recon_weight=0 (and may
freeze the VQ encoder/decoder) → AR training on (IDS, refs) → target tokens.
"""

from __future__ import annotations

import dataclasses
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from paper_reimpl_shared.config import resolve_path
from paper_reimpl_shared.data.manifest import BackendPaths
from torch.utils.data import DataLoader

from .dataset import IFFontCollate, build_dataset, load_ids_lookup
from .ids import IDSTokenizer
from .model import IFFont, IFFontConfig, VQTokenizerConfig, build_if_font

__all__ = ["compute_loss", "main"]


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------


def compute_loss(
    *,
    model: IFFont,
    batch: dict[str, torch.Tensor],
    ce_weight: float = 1.0,
    vq_weight: float = 1.0,
    recon_weight: float = 1.0,
    cfg_drop_prob: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute IF-Font's composite loss.

    Args:
        model: ``IFFont`` instance.
        batch: dict with keys ``image``, ``refs`` (or ``ref_images``), and
            optional ``ids_token_ids`` + ``ids_attention_mask``.
        ce_weight, vq_weight, recon_weight: scalar weights.
        cfg_drop_prob: probability of dropping the IDS conditioning per
            sample (classifier-free guidance training). Refs are kept.
    """
    image = batch["image"]
    ref_images = batch.get("refs") if "refs" in batch else batch.get("ref_images")
    if ref_images is not None and ref_images.numel() == 0:
        ref_images = None
    ids_token_ids = batch.get("ids_token_ids")
    ids_attention_mask = batch.get("ids_attention_mask")

    if cfg_drop_prob > 0.0 and ids_attention_mask is not None and ids_token_ids is not None:
        b = ids_attention_mask.shape[0]
        drop = torch.rand(b, device=ids_attention_mask.device) < cfg_drop_prob
        # Zeroing the attention mask effectively makes the IDS branch a no-op
        # for the dropped rows (the masked-softmax will not be used).
        ids_attention_mask = ids_attention_mask.clone()
        ids_attention_mask[drop, :] = False

    out = model(
        target_image=image,
        ids_token_ids=ids_token_ids,
        ids_attention_mask=ids_attention_mask,
        ref_images=ref_images,
    )

    logits = out["logits"]  # [B, N, K]
    target_ids = out["target_ids"]  # [B, N]

    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="mean",
    )
    vq = out["vq_loss"]
    rec = out["recon_loss"]
    total = ce_weight * ce + vq_weight * vq + recon_weight * rec
    log = {
        "loss_total": float(total.detach().cpu()),
        "loss_ce": float(ce.detach().cpu()),
        "loss_vq": float(vq.detach().cpu()),
        "loss_recon": float(rec.detach().cpu()),
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
        in_channels=int(m.get("in_channels", 1)),
        base_channels=int(vq.get("base_channels", 64)),
        channel_mult=tuple(int(x) for x in vq.get("channel_mult", [1, 2, 2, 4])),
        embedding_dim=int(vq.get("embedding_dim", 256)),
        codebook_size=int(vq.get("codebook_size", 256)),
        commitment_weight=float(vq.get("commitment_weight", 0.25)),
        decay=float(vq.get("decay", 0.99)),
    )


def _model_cfg_from_yaml(model_cfg: dict[str, Any]) -> IFFontConfig:
    m = model_cfg.get("model", model_cfg)
    vq_cfg = _vq_cfg_from_yaml(m)
    return IFFontConfig(
        image_size=int(m.get("image_size", 128)),
        in_channels=int(m.get("in_channels", 1)),
        vq=vq_cfg,
        ids_vocab_size=int(m.get("ids_vocab_size", 1024)),
        ids_max_len=int(m.get("ids_max_len", 32)),
        ids_encoder_layers=int(m.get("ids_encoder_layers", 2)),
        ids_encoder_heads=int(m.get("ids_encoder_heads", 4)),
        ids_encoder_dim=int(m.get("ids_encoder_dim", 384)),
        d_model=int(m.get("d_model", 384)),
        n_heads=int(m.get("n_heads", 8)),
        n_blocks=int(m.get("n_blocks", 10)),
        n_self_attn_per_block=int(m.get("n_self_attn_per_block", 2)),
        ffn_mult=int(m.get("ffn_mult", 4)),
        dropout=float(m.get("dropout", 0.0)),
        n_refs=int(m.get("n_refs", 1)),
    )


def _warm_fit_tokenizer(
    tokenizer: IDSTokenizer,
    dataset,
    *,
    max_samples: int | None = None,
) -> None:
    """Pre-scan the dataset to populate the tokenizer vocab, then freeze.

    This avoids the data-race where ``IFFontCollate._maybe_fit`` mutates the
    tokenizer inside a DataLoader worker subprocess — the mutation would
    never reach the main process. Pre-fitting + freezing is the correct
    pattern for ``num_workers > 0`` runs.

    The warm pass uses each row's ``ids_string`` field (added either by
    ``IFFontDataset._fetch`` or ``_SyntheticIFFontDataset.__getitem__``).
    Falls back to a per-index dataset iteration; we keep this simple and
    sequential since it runs once at startup.
    """
    n = len(dataset)
    if max_samples is not None:
        n = min(n, int(max_samples))
    ids_strings: list[str] = []
    for i in range(n):
        try:
            row = dataset[i]
        except Exception:  # pragma: no cover — degraded mode for sparse datasets
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
) -> DataLoader:
    ids_lookup = load_ids_lookup(data_cfg.get("ids_lookup_path"))
    dataset = build_dataset(
        args=args,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        paths=paths,
        ids_lookup=ids_lookup,
    )
    bs = int(train_cfg.get("batch_size", 4))
    nw = int(train_cfg.get("num_workers", 0))
    if getattr(args, "dry_run", False):
        nw = 0

    # Warm-fit the tokenizer BEFORE the DataLoader is constructed so that
    # `num_workers > 0` is safe (worker-side vocab growth would not reach
    # the main process). If the tokenizer is already frozen by the caller,
    # _warm_fit_tokenizer is a no-op.
    if not tokenizer.is_frozen:
        _warm_fit_tokenizer(
            tokenizer,
            dataset,
            max_samples=int(train_cfg.get("tokenizer_warm_fit_max_samples", 0)) or None,
        )

    collate = IFFontCollate(
        tokenizer=tokenizer,
        max_refs=int(data_cfg.get("max_refs", 1)),
        ids_max_len=int(model_cfg.ids_max_len),
        fit_on_first_call=False,
    )
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=False,
        num_workers=nw,
        collate_fn=collate,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def main(args, *, data_cfg, model_cfg, train_cfg, paths: BackendPaths) -> int:
    """Entrypoint dispatched from paper_reimpl_shared.runner.entrypoint."""
    device = torch.device(args.device)
    _seed_everything(int(train_cfg.get("seed", 42)))

    cfg = _model_cfg_from_yaml(model_cfg)

    # Build tokenizer with the IDC-only vocab. The DataLoader builder runs
    # a warm fit pass over the dataset's IDS strings BEFORE constructing
    # the DataLoader and then freezes the tokenizer, which keeps multi-
    # worker training safe (worker-side mutation would not reach the main
    # process). Paper specifies IDS but not the dict source — we use the
    # CHISE-derived CNS table via ids_lookup_path.
    tokenizer = IDSTokenizer.from_idc_only()
    # Pad the model vocab so the IDS embedding table can fit the warm-fit
    # vocab size with headroom for occasional manifest churn.
    _IDS_VOCAB_HEADROOM = 4096  # upper bound on unique CJK leaf components
    cfg.ids_vocab_size = max(cfg.ids_vocab_size, tokenizer.vocab_size + _IDS_VOCAB_HEADROOM)

    model = build_if_font(cfg).to(device)

    loader = _build_dataloader(
        args=args,
        data_cfg=data_cfg,
        model_cfg=cfg,
        train_cfg=train_cfg,
        paths=paths,
        tokenizer=tokenizer,
    )
    # Sanity check: after warm-fit, the tokenizer must still fit the
    # model's embedding table.
    if tokenizer.vocab_size > cfg.ids_vocab_size:
        raise ValueError(
            f"IDS tokenizer vocab ({tokenizer.vocab_size}) exceeds model "
            f"ids_vocab_size ({cfg.ids_vocab_size}); increase vocab headroom."
        )

    lr = float(train_cfg.get("learning_rate", 2.0e-4))
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    cfg_drop = float(train_cfg.get("cfg_drop_prob", 0.0))
    ce_weight = float(train_cfg.get("ce_weight", 1.0))
    vq_weight = float(train_cfg.get("vq_weight", 1.0))
    recon_weight = float(train_cfg.get("recon_weight", 1.0))

    # Freeze the VQ codebook outside Stage A. In Stage B/C the AR target is
    # the codebook indices; if the codebook EMA keeps updating, the CE target
    # drifts under the AR objective. Defaults: freeze when both vq_weight and
    # recon_weight are zero (the canonical Stage B/C config); explicit
    # `freeze_codebook` in train YAML overrides.
    default_freeze = (vq_weight == 0.0 and recon_weight == 0.0)
    freeze_codebook = bool(train_cfg.get("freeze_codebook", default_freeze))
    model.vq.quantizer.update_codebook = not freeze_codebook
    max_steps = int(train_cfg.get("max_steps", 1 if args.dry_run else 1_000_000))
    log_every = int(train_cfg.get("log_every", 50))

    ckpt_dir = train_cfg.get("ckpt_dir")
    if ckpt_dir is not None:
        ckpt_dir = resolve_path(ckpt_dir, base=Path(__file__).resolve().parents[3])
        os.makedirs(ckpt_dir, exist_ok=True)

    print(
        f"[if_font] device={device} steps={max_steps} bs={train_cfg.get('batch_size')} "
        f"lr={lr} ce={ce_weight} vq={vq_weight} recon={recon_weight} cfg_drop={cfg_drop} "
        f"codebook={cfg.vq.codebook_size} d_model={cfg.d_model} blocks={cfg.n_blocks} "
        f"freeze_codebook={freeze_codebook}"
    )

    model.train()
    step = 0
    for _ in range(int(train_cfg.get("max_epochs", 1))):
        for batch in loader:
            batch = _move_batch(batch, device)
            optim.zero_grad(set_to_none=True)
            loss, log = compute_loss(
                model=model,
                batch=batch,
                ce_weight=ce_weight,
                vq_weight=vq_weight,
                recon_weight=recon_weight,
                cfg_drop_prob=cfg_drop,
            )
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            if step % log_every == 0:
                print(
                    f"[if_font] step={step} total={log['loss_total']:.4f} "
                    f"ce={log['loss_ce']:.4f} vq={log['loss_vq']:.4f} "
                    f"recon={log['loss_recon']:.4f}"
                )
            step += 1
            if step >= max_steps or args.dry_run:
                break
        if step >= max_steps or args.dry_run:
            break

    if ckpt_dir is not None and not args.dry_run:
        path = Path(ckpt_dir) / "if_font_last.pt"
        # Use dataclasses.asdict to deep-convert the nested VQTokenizerConfig
        # into a plain dict — cfg.__dict__ would leave the nested dataclass
        # as a live object, which is brittle across class-definition changes.
        torch.save({"model": model.state_dict(), "cfg": dataclasses.asdict(cfg)}, path)
        print(f"[if_font] saved checkpoint -> {path}")

    print(f"[if_font] done; final_step={step} dry_run={args.dry_run}")
    return 0
