"""DP-Font sampling helpers.

Thin wrappers around ``paper_reimpl_shared.diffusion.gaussian.GaussianDiffusion``
that bake in DP-Font's multi-attribute conditioning convention.

The shared sampler does not natively dispatch DP-Font's extra kwargs
(``stroke_order``, ``ink_intensity``, ``font_size``). We work around this by
wrapping the DP-Font model in a small "frozen-condition" adapter that
captures those fields once at sample-time and forwards them on every call —
this keeps the conditioning identical to the training distribution without
having to extend the shared sampler API.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion

from .model import DPFont


__all__ = ["sample_ddpm", "sample_ddim", "sample"]


class _FrozenCondAdapter(nn.Module):
    """Wrap a DPFont so the shared sampler's forward kwargs route correctly.

    Captures the optional DP-Font-specific kwargs (``stroke_order``,
    ``ink_intensity``, ``font_size``) once on construction; passes them
    through on every call. The shared sampler only sees the standard model
    signature, so it can still drop categorical attributes for CFG via the
    ``cfg_uncond_drops_content`` / ``char_id=None`` machinery.
    """

    def __init__(
        self,
        model: DPFont,
        *,
        stroke_order: torch.Tensor | None,
        ink_intensity: torch.Tensor | None,
        font_size: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.model = model
        self.stroke_order = stroke_order
        self.ink_intensity = ink_intensity
        self.font_size = font_size

    def forward(self, x_t, timesteps, **kwargs):  # type: ignore[no-untyped-def]
        return self.model(
            x_t,
            timesteps,
            stroke_order=self.stroke_order,
            ink_intensity=self.ink_intensity,
            font_size=self.font_size,
            **kwargs,
        )


@torch.no_grad()
def sample(
    *,
    model: DPFont,
    diffusion: GaussianDiffusion,
    content: torch.Tensor,
    writer_id: torch.Tensor | None = None,
    script_id: torch.Tensor | None = None,
    char_id: torch.Tensor | None = None,
    stroke_order: torch.Tensor | None = None,
    ink_intensity: torch.Tensor | None = None,
    font_size: torch.Tensor | None = None,
    sampler: str = "ddpm",
    cfg_scale: float = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate glyphs conditioned on the multi-attribute fields.

    Returns:
        [B, 1, H, W] tensor in [-1, 1].
    """
    device = torch.device(device) if device is not None else content.device
    content = content.to(device)
    b = content.shape[0]
    image_size = model.cfg.image_size
    in_channels = model.cfg.in_channels

    adapter = _FrozenCondAdapter(
        model,
        stroke_order=stroke_order.to(device) if stroke_order is not None else None,
        ink_intensity=ink_intensity.to(device) if ink_intensity is not None else None,
        font_size=font_size.to(device) if font_size is not None else None,
    )

    return diffusion.sample(
        adapter,
        shape=(b, in_channels, image_size, image_size),
        content=content,
        char_id=char_id.to(device) if char_id is not None else None,
        script_id=script_id.to(device) if script_id is not None else None,
        writer_id=writer_id.to(device) if writer_id is not None else None,
        sampler=sampler,
        cfg_scale=cfg_scale,
        # The DP-Font training drops categorical attributes (not content) for
        # CFG, so the uncond branch should keep content (only categorical
        # ids are nulled by the shared sampler when cfg_scale != 1.0).
        cfg_uncond_drops_content=False,
        device=device,
    )


def sample_ddpm(
    *,
    model: DPFont,
    diffusion: GaussianDiffusion,
    content: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return sample(model=model, diffusion=diffusion, content=content, sampler="ddpm", **kwargs)


def sample_ddim(
    *,
    model: DPFont,
    diffusion: GaussianDiffusion,
    content: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return sample(model=model, diffusion=diffusion, content=content, sampler="ddim", **kwargs)
