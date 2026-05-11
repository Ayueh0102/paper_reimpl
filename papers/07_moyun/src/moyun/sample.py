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

import warnings
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
    """Channels of the diffusion input. Defaults to 4 (VAE-latent / production
    path). The smoke / Stage-A direct-pixel path uses 1. The
    ``model.cfg.in_channels`` of the actual model is the source of truth —
    use :meth:`validate` to surface a warning if they disagree."""
    sampler: str = "ddpm"  # "ddpm" or "ddim"
    cfg_scale: float = 1.5
    seed: int = 0

    def validate(self, model: Moyun) -> None:
        """Warn if this config's ``in_channels`` disagrees with ``model``.

        ``MoyunSampleConfig.in_channels`` defaults to 4 (latent mode) while
        the smoke / Stage-A path uses ``in_channels=1``. The model's
        ``cfg.in_channels`` is the source of truth — a mismatch here would
        produce a shape error inside ``PatchEmbed``; this validate() catches
        it earlier with a clearer message.
        """
        model_ic = int(model.cfg.in_channels)
        if int(self.in_channels) != model_ic:
            warnings.warn(
                f"MoyunSampleConfig.in_channels={self.in_channels} disagrees "
                f"with model.cfg.in_channels={model_ic}; the model's value is "
                f"authoritative for shape inference. Set sample_cfg.in_channels "
                f"= model.cfg.in_channels to silence this warning.",
                stacklevel=2,
            )


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
    cfg.validate(model)
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

    # Reproducibility: the shared ``GaussianDiffusion.sample`` does NOT accept
    # a ``torch.Generator`` — its initial-noise ``randn`` and per-step
    # ``randn`` calls draw from the global default RNG. We seed that RNG
    # (both CPU and CUDA) here so ``cfg.seed`` is honoured end-to-end. Note
    # that ``init_image`` on the shared sampler is interpreted as a CLEAN
    # image to which ``q_sample`` adds noise at ``start_timestep`` — it is
    # not a slot for raw initial noise, so we cannot use it as the seed
    # plumbing. If multi-process determinism is later required, plumb a
    # ``generator`` kwarg through ``GaussianDiffusion.sample``.
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

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
