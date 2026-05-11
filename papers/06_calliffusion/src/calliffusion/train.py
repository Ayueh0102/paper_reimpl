"""Calliffusion training entry called by the unified runner.

Supports two data backends:
  - real JSONL manifests via shared ``CalligraphyJsonlDataset``
  - synthetic tensors via ``SyntheticPromptDataset`` (used with ``--synthetic``)

Stages are differentiated only by the ``train_cfg`` YAML (Stage A/B freezes
BERT; Stage C wraps cross-attention with LoRA). The model graph is the same.

This module exposes ``main(args, *, data_cfg, model_cfg, train_cfg, paths)``
matching the contract in ``paper_reimpl_shared.runner.entrypoint``.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion
from torch.utils.data import DataLoader, Dataset

from .dataset import (
    CalliffusionPromptDataset,
    SyntheticPromptDataset,
    collate_prompt_batch,
)
from .lora import apply_lora_to_module, freeze_non_lora, lora_parameters
from .model import SpatialCrossAttention, build_unet_from_yaml
from .text import build_text_encoder

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(
    *,
    synthetic: bool,
    data_cfg: dict[str, Any],
    manifest_root: Path,
    manifest_override: str | None = None,
) -> Dataset:
    image_size = int(data_cfg.get("image_size", 64))
    if synthetic:
        return SyntheticPromptDataset(
            length=int(data_cfg.get("synthetic_length", 16)),
            image_size=image_size,
            writer_vocab_size=int(data_cfg.get("synthetic_writers", 4)),
            char_vocab_size=int(data_cfg.get("synthetic_chars", 16)),
            prompt_dropout_p=float(data_cfg.get("prompt_dropout_p", 0.1)),
            seed=int(data_cfg.get("seed", 0)),
        )
    manifest_name = manifest_override or data_cfg.get("manifest")
    if not manifest_name:
        raise ValueError("data_cfg.manifest is required for non-synthetic runs")
    manifest_path = Path(manifest_root) / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    return CalliffusionPromptDataset(
        str(manifest_path),
        image_size=image_size,
        content_channels=list(data_cfg.get("content_channels", [])),
        max_refs=int(data_cfg.get("max_refs", 0)),
        prompt_dropout_p=float(data_cfg.get("prompt_dropout_p", 0.1)),
        seed=int(data_cfg.get("seed", 0)),
    )


def build_text_encoder_from_cfg(model_cfg: dict[str, Any]) -> nn.Module:
    section = model_cfg.get("text_encoder", {})
    return build_text_encoder(
        use_bert=bool(section.get("use_bert", False)),
        hidden_size=int(section.get("hidden_size", 768)),
        max_length=int(section.get("max_length", 32)),
        model_name=str(section.get("model_name", "bert-base-chinese")),
        cache_dir=section.get("cache_dir"),
    )


def maybe_apply_lora(model: torch.nn.Module, train_cfg: dict[str, Any]) -> int:
    lora_cfg = train_cfg.get("lora", {})
    if not lora_cfg.get("enabled", False):
        return 0
    # Scope LoRA to cross-attention only. Without this, the ``to_out``
    # substring would also match ``SpatialSelfAttention.to_out`` and
    # silently inflate the trainable parameter count beyond the
    # paper-stated "cross-attention projections only" target.
    wrapped = apply_lora_to_module(
        model,
        target_substrings=tuple(lora_cfg.get("target_substrings", ["to_q", "to_k", "to_v", "to_out"])),
        parent_types=(SpatialCrossAttention,),
        rank=int(lora_cfg.get("rank", 4)),
        alpha=float(lora_cfg.get("alpha", 8.0)),
        dropout=float(lora_cfg.get("dropout", 0.0)),
    )
    if lora_cfg.get("freeze_non_lora", True):
        freeze_non_lora(model)
    return wrapped


def freeze_text_encoder(text_encoder: nn.Module, train_cfg: dict[str, Any]) -> None:
    section = train_cfg.get("text_encoder", {})
    if not section.get("freeze", True):
        return
    if hasattr(text_encoder, "freeze"):
        text_encoder.freeze(embeddings_trainable=bool(section.get("embeddings_trainable", False)))
    else:
        for p in text_encoder.parameters():
            p.requires_grad = False
        if section.get("embeddings_trainable", False) and hasattr(text_encoder, "embedding"):
            for p in text_encoder.embedding.parameters():
                p.requires_grad = True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(
    args,
    *,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    paths,
) -> int:
    device = torch.device(args.device)
    set_seed(int(train_cfg.get("seed", 42)))

    # ------------------------------------------------------------------ data
    dataset = build_dataset(
        synthetic=bool(args.synthetic),
        data_cfg=data_cfg,
        manifest_root=Path(paths.manifest_root),
        manifest_override=args.manifest,
    )
    batch_size = int(train_cfg.get("batch_size", 2))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not args.dry_run,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_prompt_batch,
        drop_last=False,
    )

    # ------------------------------------------------------------------ model
    unet = build_unet_from_yaml(model_cfg).to(device)
    text_encoder = build_text_encoder_from_cfg(model_cfg).to(device)

    # If the dataset exposes writer names, register them as special tokens so
    # the encoder doesn't shred them. Real BERT does this lossy by default —
    # see paper_notes/06.md §2.4.
    writer_names: list[str] = []
    if hasattr(dataset, "writer_names"):
        writer_names = list(dataset.writer_names())
    if writer_names and hasattr(text_encoder, "add_special_tokens"):
        added = text_encoder.add_special_tokens(writer_names)
        print(f"[calliffusion] registered {added} writer special tokens")

    freeze_text_encoder(text_encoder, train_cfg)

    # Stage-C LoRA toggle
    wrapped_layers = maybe_apply_lora(unet, train_cfg)
    if wrapped_layers:
        print(f"[calliffusion] wrapped {wrapped_layers} cross-attn projections with LoRA")

    # ------------------------------------------------------------------ warm-start
    # Load weights from --init-ckpt before building the optimizer so the
    # AdamW state is created with the loaded params. Weights-only,
    # strict=False so we tolerate LoRA / text-encoder head differences
    # across stages.
    init_ckpt = getattr(args, "init_ckpt", None)
    if init_ckpt:
        blob = torch.load(init_ckpt, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        missing, unexpected = unet.load_state_dict(state, strict=False)
        print(
            f"[calliffusion] warm-start from {init_ckpt} "
            f"(missing={len(missing)} unexpected={len(unexpected)})"
        )

    # ------------------------------------------------------------------ optim
    optim_params: list[torch.nn.Parameter] = [p for p in unet.parameters() if p.requires_grad]
    if wrapped_layers:
        # When LoRA is on we already froze the base U-Net; collect just the
        # LoRA params (plus any trainable text-encoder embedding rows).
        optim_params = lora_parameters(unet)
    optim_params += [p for p in text_encoder.parameters() if p.requires_grad]
    if not optim_params:
        raise RuntimeError("No trainable parameters — check freeze + LoRA flags.")
    lr = float(train_cfg.get("lr", 1e-5))
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=lr,
        betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.999))),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    # ------------------------------------------------------------------ diff
    diffusion = GaussianDiffusion(
        timesteps=int(train_cfg.get("timesteps", 1000)),
        beta_start=float(train_cfg.get("beta_start", 1e-4)),
        beta_end=float(train_cfg.get("beta_end", 2e-2)),
        beta_schedule=str(train_cfg.get("beta_schedule", "linear")),
        prediction_target="epsilon",
        device=device,
    )

    # ------------------------------------------------------------------ loop
    max_steps = int(train_cfg.get("max_steps", 1))
    if args.dry_run:
        max_steps = 1
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = max(1, int(train_cfg.get("log_every", 10)))

    unet.train()
    # Only put the text encoder in train() mode when it has at least one
    # trainable parameter. Stage A/C freeze BERT entirely; leaving it in
    # train() would activate the BERT internal Dropout layers and inject
    # noise into what should be a deterministic feature extractor.
    if any(p.requires_grad for p in text_encoder.parameters()):
        text_encoder.train()
    else:
        text_encoder.eval()

    step = 0
    last_loss: float = float("nan")
    for epoch in range(int(train_cfg.get("max_epochs", 1000))):
        for batch in loader:
            images = batch["image"].to(device)
            prompts = batch["prompt"]
            ctx_out = text_encoder.encode(prompts) if hasattr(text_encoder, "encode") else text_encoder(prompts)
            d_batch = diffusion.sample_training_batch(images)
            pred = unet(
                d_batch.x_t,
                d_batch.timesteps,
                context=ctx_out.last_hidden_state,
                context_mask=ctx_out.attention_mask,
            )
            loss = F.mse_loss(pred, d_batch.target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss is non-finite at step {step}: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(optim_params, grad_clip)
            optimizer.step()
            last_loss = float(loss.detach().item())
            if step % log_every == 0:
                print(
                    f"[calliffusion] epoch={epoch} step={step} "
                    f"loss={last_loss:.6f} lr={lr:.2e}"
                )
            step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    print(f"[calliffusion] done. final_step={step} final_loss={last_loss:.6f}")

    if not args.dry_run:
        ckpt_dir_raw = train_cfg.get("ckpt_dir")
        if ckpt_dir_raw is not None:
            from pathlib import Path as _P
            ckpt_dir = _P(str(ckpt_dir_raw))
            if not ckpt_dir.is_absolute():
                # parents: [0]=calliffusion [1]=src [2]=06_calliffusion [3]=papers [4]=repo
                ckpt_dir = _P(__file__).resolve().parents[4] / ckpt_dir
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "calliffusion_last.pt"
            torch.save(
                {
                    "model": unet.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "lora_wrapped": wrapped_layers,
                },
                ckpt_path,
            )
            print(f"[calliffusion] saved checkpoint -> {ckpt_path}")

    if args.dry_run:
        if not math.isfinite(last_loss):
            print("[calliffusion] DRY RUN FAILED — non-finite loss")
            return 1
        print("[calliffusion] DRY RUN OK")
    return 0
