"""HFH-Font inference / sampling utilities.

Phase-1 surface:
- ``sample_latents``: run the latent-space DDPM (or DDIM trailing-K-step
  approximation) using ``paper_reimpl_shared.diffusion.GaussianDiffusion``.
- ``decode_samples``: VAE-decode latents back to glyph images.
- ``run_sr``: optional style-guided SR upsample (Phase-1 stub).
"""

from __future__ import annotations

from typing import Any

import torch
from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion

from .model import HFHFontModel

__all__ = ["sample_latents", "decode_samples", "run_sr"]


@torch.no_grad()
def sample_latents(
    model: HFHFontModel,
    diffusion: GaussianDiffusion,
    *,
    batch: dict[str, Any],
    sampler: str = "ddim",
    cfg_scale: float = 2.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Run reverse diffusion in latent space.

    The paper note specifies ``sampler=ddim``-like with 10 trailing steps and
    ``sc = ss = 2.0``. We keep the same CFG scale wiring as
    ``GaussianDiffusion.sample``.
    """
    z0_shape = _infer_latent_shape(model, batch)
    return diffusion.sample(
        model,
        shape=z0_shape,
        content=batch["content"].to(device),
        char_id=_maybe_to(batch.get("char_id"), device),
        script_id=_maybe_to(batch.get("script_id"), device),
        writer_id=_maybe_to(batch.get("writer_id"), device),
        style_family_id=_maybe_to(batch.get("style_family_id"), device),
        unit_id=_maybe_to(batch.get("unit_id"), device),
        ref_images=_maybe_to(batch.get("ref_images"), device),
        ref_valid=_maybe_to(batch.get("ref_valid"), device),
        sampler=sampler,
        cfg_scale=cfg_scale,
        cfg_uncond_drops_content=False,
        device=device,
    )


@torch.no_grad()
def decode_samples(model: HFHFontModel, latents: torch.Tensor) -> torch.Tensor:
    return model.decode_latent(latents)


@torch.no_grad()
def run_sr(model: HFHFontModel, lr_images: torch.Tensor, ref_tokens: torch.Tensor) -> torch.Tensor:
    if model.sr_module is None:
        raise RuntimeError("SR module disabled in config; set sr_enabled=true to use run_sr")
    return model.sr_module(lr_images, ref_tokens)


def _infer_latent_shape(model: HFHFontModel, batch: dict[str, Any]) -> tuple[int, int, int, int]:
    image = batch["image"]
    f = model.cfg.vae_down_factor
    return (image.shape[0], model.cfg.latent_channels, image.shape[-2] // f, image.shape[-1] // f)


def _maybe_to(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value
