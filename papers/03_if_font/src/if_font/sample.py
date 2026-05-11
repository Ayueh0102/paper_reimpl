"""IF-Font inference (autoregressive sampling) — Phase 2.

Usage (programmatic):
    from if_font import IFFont, IDSTokenizer
    out = if_font_sample(model, tokenizer, ids_strings, ref_images, coverage_sim)

IF-Font Phase 2 requires (ids_token_ids, ref_images, coverage_sim) — the
coverage signal is what drives StyleEncoder's ref-routing.
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
    ids_strings: Sequence[str | Sequence[str]],
    ref_images: torch.Tensor,
    coverage_sim: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = 100,
    sample: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate target glyphs autoregressively.

    Args:
        model: a trained ``IFFont`` (Phase-2 model).
        tokenizer: the ``IDSTokenizer`` whose vocab the model was trained
            against.
        ids_strings: per-sample IDS strings (or pre-tokenised tuples).
        ref_images: [B, N, C, H, W] reference glyphs in [-1, 1].
        coverage_sim: [B, N] float — coverage similarity per ref.
        temperature, top_k, sample: AR sampling controls.
        device: torch device for the run.
    """
    model.eval()
    device = torch.device(device)
    ids_token_ids, _ = tokenizer.batch_encode(
        list(ids_strings), max_len=model.cfg.ids_max_len
    )
    ids_token_ids = ids_token_ids.to(device)
    refs = ref_images.to(device)
    cov = coverage_sim.to(device)
    return model.sample(
        ids_token_ids=ids_token_ids,
        ref_images=refs,
        coverage_sim=cov,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
    )
