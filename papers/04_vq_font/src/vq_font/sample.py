"""VQ-Font sampling — recipes for inference time.

Two top-level entrypoints:

* :func:`sample_vqgan_recon` — round-trip an image through VQGAN
  (encode → quantize → decode). Useful for visualising what the codebook
  alone can reconstruct at the end of Stage 0.
* :func:`sample_vq_font` — full Stage 1+ pipeline: take an initial
  synthesized glyph + R reference glyphs + structure id, predict codebook
  indices via the Transformer, decode through VQGAN. Supports two
  decoding modes:
      ``argmax``  — deterministic, fastest, paper default.
      ``sample``  — temperature-controlled categorical sampling for diverse
                    outputs (`temperature` and `top_k`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import VQFont
from .vqgan import VQGAN

__all__ = ["sample_vqgan_recon", "sample_vq_font", "indices_from_logits"]


@torch.no_grad()
def sample_vqgan_recon(model: VQGAN, x: torch.Tensor) -> torch.Tensor:
    """Encode + quantize + decode. Used to inspect Stage-0 reconstruction quality."""
    out = model(x)
    return out.recon


def indices_from_logits(
    logits: torch.Tensor,
    *,
    mode: str = "argmax",
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Convert per-token codebook logits [B, N, K] into integer indices [B, N].

    Args:
        logits: shape [B, N, K].
        mode: 'argmax' or 'sample'.
        temperature: softmax temperature (only relevant for 'sample').
        top_k: keep only top-k entries before softmax (only relevant for 'sample').
    """
    if mode == "argmax":
        return logits.argmax(dim=-1)
    if mode != "sample":
        raise ValueError(f"Unknown mode {mode!r}")
    b, n, k = logits.shape
    z = logits / max(float(temperature), 1e-6)
    if top_k is not None and top_k > 0 and top_k < k:
        topv, topi = z.topk(top_k, dim=-1)
        mask = torch.full_like(z, float("-inf"))
        mask.scatter_(-1, topi, topv)
        z = mask
    probs = F.softmax(z, dim=-1).reshape(b * n, k)
    idx = torch.multinomial(probs, num_samples=1).reshape(b, n)
    return idx


@torch.no_grad()
def sample_vq_font(
    *,
    model: VQFont,
    initial_glyph: torch.Tensor,
    ref_glyphs: torch.Tensor,
    structure_id: torch.Tensor,
    ref_valid: torch.Tensor | None = None,
    mode: str = "argmax",
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Run the full Token Prior Refinement pipeline.

    Args:
        model: trained ``VQFont`` (frozen VQGAN + trained Transformer).
        initial_glyph: [B, C, H, W] initial synthesis (or source glyph).
        ref_glyphs:    [B, R, C, H, W].
        structure_id:  [B] long structure class id (0..13).
        ref_valid:     optional [B, R] bool.
        mode/temperature/top_k: see :func:`indices_from_logits`.

    Returns:
        [B, C, H, W] reconstructed glyph image in [-1, 1] (decoder output).
    """
    token_logits, _ = model.predict_token_logits(
        initial_glyph, ref_glyphs, structure_id, ref_valid=ref_valid
    )
    b = token_logits.shape[0]
    flat_idx = indices_from_logits(token_logits, mode=mode, temperature=temperature, top_k=top_k)
    lat = model.cfg.transformer.latent_resolution
    idx_grid = flat_idx.reshape(b, lat, lat)
    return model.decode_indices_to_image(idx_grid)
