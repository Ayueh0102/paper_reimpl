"""IF-Font inference (autoregressive sampling).

Usage (programmatic):
    from if_font import IFFont, IDSTokenizer
    out = if_font_sample(model, tokenizer, ids_strings, ref_images)

The shared sampler API uses GaussianDiffusion; IF-Font is autoregressive so
we expose a dedicated entry point. The shared
``paper_reimpl_shared.runner.entrypoint`` is the trainer entry — sampling
goes through this module directly (called by future eval scripts).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .ids import IDSTokenizer
from .model import IFFont

__all__ = ["if_font_sample"]


@torch.no_grad()
def if_font_sample(
    model: IFFont,
    tokenizer: IDSTokenizer,
    ids_strings: Sequence[str],
    ref_images: torch.Tensor | None,
    *,
    temperature: float = 1.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate target glyphs autoregressively.

    Args:
        model: a trained ``IFFont`` (or freshly initialised — produces noise).
        tokenizer: the ``IDSTokenizer`` whose vocab the model was trained
            against.
        ids_strings: per-sample IDS strings (e.g. ``"⿰示畐"`` for 福).
        ref_images: [B, N, C, H, W] reference glyphs in [-1, 1]; ``None`` for
            zero-shot (paper's source-glyph-free claim is exercised here).
        temperature: AR sampling temperature; 0 = greedy.
        device: torch device for the run.
    """
    model.eval()
    device = torch.device(device)
    ids_token_ids, ids_attention_mask = tokenizer.batch_encode(
        list(ids_strings), max_len=model.cfg.ids_max_len
    )
    ids_token_ids = ids_token_ids.to(device)
    ids_attention_mask = ids_attention_mask.to(device)
    refs = ref_images.to(device) if isinstance(ref_images, torch.Tensor) else None
    return model.sample(
        ids_token_ids=ids_token_ids,
        ids_attention_mask=ids_attention_mask,
        ref_images=refs,
        temperature=temperature,
    )
