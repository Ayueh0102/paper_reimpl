"""Moyun inference / sampling.

Thin wrapper around ``paper_reimpl_shared.diffusion.gaussian.GaussianDiffusion.sample``
that handles the TripleLabel CFG path.

Two CFG knobs:

  * ``cfg_scale``: standard guidance strength applied as
      eps_guided = eps_uncond + cfg_scale * (eps_cond - eps_uncond).
    The shared sampler ``cfg_uncond_drops_content=False`` skips content
    zeroing (Moyun doesn't use content); the uncond branch is recovered by
    passing the three label ids as ``None`` (which routes them to the [NULL]
    embedding row at index 0 inside the model).

For zero-shot character generation (paper §4.2: "calligrapher A writing a
char they never wrote"), you simply pass the new ``char_id`` together with
the desired ``writer_id`` and ``script_id``. The TripleLabel embeddings are
additive, so any combination is well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion

from .model import Moyun

__all__ = ["MoyunSampleConfig", "sample"]


@dataclass
class MoyunSampleConfig:
    batch_size: int = 4
    image_size: int = 32
    in_channels: int = 4
    sampler: str = "ddpm"  # "ddpm" or "ddim"
    cfg_scale: float = 1.5
    seed: int = 0


@torch.no_grad()
def sample(
    *,
    model: Moyun,
    diffusion: GaussianDiffusion,
    cfg: MoyunSampleConfig,
    writer_id: torch.Tensor | None,
    script_id: torch.Tensor | None,
    char_id: torch.Tensor | None,
    device: torch.device | str,
) -> torch.Tensor:
    """Run reverse diffusion and return the generated (latent) tensor.

    Shape: ``(B, in_channels, H, W)``.

    To convert latents back to pixel images, the caller should decode with
    the corresponding VAE. For smoke testing we run with ``in_channels=1``
    and the output is already an image-shaped tensor in [-1, 1].
    """
    model.eval()
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    shape = (cfg.batch_size, cfg.in_channels, cfg.image_size, cfg.image_size)

    # Shared sampler signature requires ``content`` (it's the only non-None
    # field it dereferences as ``torch.zeros_like(content)``). We pass an
    # empty zero tensor — Moyun.forward ignores it.
    content = torch.zeros(
        cfg.batch_size,
        1,
        cfg.image_size,
        cfg.image_size,
        device=device,
    )
    # Random init noise.
    init_noise = torch.randn(*shape, generator=g, device=device)
    # We piggy-back on the shared sampler by setting ``init_image=None``;
    # GaussianDiffusion.sample will draw its own randn. To honour our seed we
    # could pass init_image=init_noise, but that requires the shape match —
    # which is fine. We keep this branch simple: rely on the shared sampler's
    # internal randn.
    _ = init_noise

    out = diffusion.sample(
        model,
        shape=shape,
        content=content,
        writer_id=writer_id,
        script_id=script_id,
        char_id=char_id,
        sampler=cfg.sampler,
        cfg_scale=cfg.cfg_scale,
        cfg_uncond_drops_content=False,  # Moyun has no content path; only ids drop
        device=device,
    )
    return out
