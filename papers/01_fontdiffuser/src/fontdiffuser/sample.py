"""FontDiffuser sampling helpers.

Thin wrappers around ``paper_reimpl_shared.diffusion.gaussian.GaussianDiffusion``
that bake in FontDiffuser's one-shot conditioning convention.

Two sampling modes:
  * ``sample_ddpm`` — full reverse diffusion (T steps), per paper.
  * ``sample_ddim`` — accelerated sampler available from the shared utility;
    convenient for qualitative checks during training.
"""

from __future__ import annotations

import torch

from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion

from .model import FontDiffuser

__all__ = ["sample_ddpm", "sample_ddim", "sample"]


def _ref_inputs(
    ref_image: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if ref_image is None:
        return None, None
    if ref_image.ndim == 4:
        # [B, C, H, W] -> [B, 1, C, H, W]
        ref_image = ref_image.unsqueeze(1)
    elif ref_image.ndim != 5:
        raise ValueError(
            f"ref_image must be [B,C,H,W] or [B,N,C,H,W]; got shape {tuple(ref_image.shape)}"
        )
    if ref_image.shape[0] != batch_size:
        raise ValueError("ref_image batch dim does not match `content` batch dim")
    ref_valid = torch.ones(ref_image.shape[0], ref_image.shape[1], dtype=torch.bool, device=device)
    return ref_image.to(device), ref_valid


@torch.no_grad()
def sample(
    *,
    model: FontDiffuser,
    diffusion: GaussianDiffusion,
    content: torch.Tensor,
    ref_image: torch.Tensor | None = None,
    sampler: str = "ddpm",
    cfg_scale: float = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate samples conditioned on a source content image + 1 style ref.

    Args:
        model: FontDiffuser instance.
        diffusion: shared GaussianDiffusion (same instance used at train time).
        content: [B, content_C, H, W] source-glyph render.
        ref_image: [B, ref_C, H, W] or [B, 1, ref_C, H, W] one-shot style
            reference. When None, the model uses its learned null style.
        sampler: 'ddpm' (full T-step) or 'ddim'.
        cfg_scale: classifier-free guidance scale relative to the
            null-reference unconditional path. cfg_scale == 1.0 disables CFG.
        device: target device; falls back to ``content.device``.
    """
    device = torch.device(device) if device is not None else content.device
    content = content.to(device)
    b = content.shape[0]
    image_size = model.cfg.image_size
    in_channels = model.cfg.in_channels
    refs, refs_valid = _ref_inputs(ref_image, b, device)
    return diffusion.sample(
        model,
        shape=(b, in_channels, image_size, image_size),
        content=content,
        ref_images=refs,
        ref_valid=refs_valid,
        sampler=sampler,
        cfg_scale=cfg_scale,
        # FontDiffuser training only drops the style ref (not content), so the
        # CFG uncond branch must keep content; otherwise cfg_scale>1.0 pulls
        # toward an OOD (zero-content, no-ref) prediction never seen during
        # training. Fix applied per DL review 2026-05-11.
        cfg_uncond_drops_content=False,
        device=device,
    )


def sample_ddpm(
    *,
    model: FontDiffuser,
    diffusion: GaussianDiffusion,
    content: torch.Tensor,
    ref_image: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    return sample(
        model=model,
        diffusion=diffusion,
        content=content,
        ref_image=ref_image,
        sampler="ddpm",
        cfg_scale=cfg_scale,
        device=device,
    )


def sample_ddim(
    *,
    model: FontDiffuser,
    diffusion: GaussianDiffusion,
    content: torch.Tensor,
    ref_image: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    return sample(
        model=model,
        diffusion=diffusion,
        content=content,
        ref_image=ref_image,
        sampler="ddim",
        cfg_scale=cfg_scale,
        device=device,
    )
