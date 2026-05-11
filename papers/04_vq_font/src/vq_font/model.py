"""Top-level VQ-Font composite model.

Phase 2 changes vs blind-impl
-----------------------------
* Drop ``StructureEncoder`` / ``StructureHead`` (now ``[blind-impl-divergence]``
  rather than ``[paper-cited]``). SSEM is now the parameter-free
  ``RegionAttentionRecalibrator`` inside ``TokenPriorTransformer``.
* ``freeze_vqgan=True`` now applies a **partial freeze** matching
  ``generator.py:40-49``: encoder + codebook + late decoder are frozen,
  the first three decoder ``ResBlock`` ``conv1/conv2`` weights stay
  trainable along with ``post_quant`` (= official ``post_quant_conv``).
  Pass ``freeze_vqgan='full'`` for the strict blind-impl behaviour
  (everything frozen) — kept so legacy tests still pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .transformer import (
    NUM_STRUCTURE_CLASSES,
    RegionAttentionRecalibrator,
    TokenPriorTransformer,
    TransformerConfig,
    build_transformer,
)
from .vqgan import (
    VQGAN,
    VQGANConfig,
    VQGANOutputs,
    VectorQuantize,
    build_vqgan,
)

__all__ = [
    "NUM_STRUCTURE_CLASSES",
    "RegionAttentionRecalibrator",
    "TokenPriorTransformer",
    "TransformerConfig",
    "VQGAN",
    "VQGANConfig",
    "VQGANOutputs",
    "VQFontConfig",
    "VQFont",
    "VectorQuantize",
    "build_transformer",
    "build_vq_font",
    "build_vqgan",
]


FreezeMode = Literal["partial", "full", "none"]


@dataclass
class VQFontConfig:
    """Combined config used at Stage 1+ where both pieces co-exist."""

    vqgan: VQGANConfig
    transformer: TransformerConfig

    def __post_init__(self) -> None:
        # Lightweight integrity checks. The transformer's latent grid must
        # match what VQGAN actually produces, and codebook sizes must agree.
        if self.transformer.codebook_size != self.vqgan.num_embeddings:
            raise ValueError(
                "VQFontConfig: transformer.codebook_size must equal "
                f"vqgan.num_embeddings; got {self.transformer.codebook_size} "
                f"vs {self.vqgan.num_embeddings}"
            )
        if self.transformer.embed_dim != self.vqgan.embed_dim:
            raise ValueError(
                "VQFontConfig: transformer.embed_dim must equal vqgan.embed_dim; "
                f"got {self.transformer.embed_dim} vs {self.vqgan.embed_dim}"
            )
        vqgan_latent = self.vqgan.out_resolution()
        if self.transformer.latent_resolution != vqgan_latent:
            raise ValueError(
                "VQFontConfig: transformer.latent_resolution must equal "
                f"vqgan.out_resolution(); got {self.transformer.latent_resolution} "
                f"vs {vqgan_latent} (vqgan.image_size={self.vqgan.image_size}, "
                f"channel_mult={self.vqgan.channel_mult})"
            )


class VQFont(nn.Module):
    """Stage-1+ composite: (partially) frozen VQGAN + trainable Transformer.

    Freeze policy is selected via the ``freeze_vqgan`` keyword on the
    constructor:

    * ``'partial'`` (default, paper-faithful, ``generator.py:40-49``):
      freeze everything in VQGAN except the first three decoder
      ``ResBlock`` ``conv1.conv``/``conv2.conv`` weights (parameter names
      ``decoder.res_blocks.{0,1,2}.conv1.conv.*`` and ``.conv2.conv.*``)
      and the ``post_quant`` projection. These are the "early decoder
      layers + post_quant_conv" that the official code keeps trainable.
    * ``'full'``: everything in VQGAN frozen (blind-impl behaviour, kept
      for the legacy smoke test).
    * ``'none'``: nothing frozen — used by Stage 0.

    ``True`` (legacy bool) maps to ``'partial'``; ``False`` maps to ``'none'``.
    """

    # Set of parameter-name suffixes that stay trainable under 'partial' freeze.
    # We match the official intent ("first three decoder ResBlock convs + post_quant_conv")
    # in our updated layer naming (``decoder.res_blocks.{0,1,2}.conv{1,2}.conv.{weight,bias}``).
    _PARTIAL_FREEZE_TRAINABLE_PATTERNS: tuple[str, ...] = (
        "decoder.res_blocks.0.conv1.conv.weight",
        "decoder.res_blocks.0.conv1.conv.bias",
        "decoder.res_blocks.0.conv2.conv.weight",
        "decoder.res_blocks.0.conv2.conv.bias",
        "decoder.res_blocks.1.conv1.conv.weight",
        "decoder.res_blocks.1.conv1.conv.bias",
        "decoder.res_blocks.1.conv2.conv.weight",
        "decoder.res_blocks.1.conv2.conv.bias",
        "decoder.res_blocks.2.conv1.conv.weight",
        "decoder.res_blocks.2.conv1.conv.bias",
        "decoder.res_blocks.2.conv2.conv.weight",
        "decoder.res_blocks.2.conv2.conv.bias",
        "post_quant.weight",
        "post_quant.bias",
    )

    def __init__(self, cfg: VQFontConfig, *, freeze_vqgan: FreezeMode | bool = "partial") -> None:
        super().__init__()
        self.cfg = cfg
        self.vqgan = build_vqgan(cfg.vqgan)
        self.transformer = build_transformer(cfg.transformer)

        # Resolve legacy bool API.
        if freeze_vqgan is True:
            mode: FreezeMode = "partial"
        elif freeze_vqgan is False:
            mode = "none"
        else:
            mode = freeze_vqgan
        self.freeze_mode: FreezeMode = mode
        self._apply_freeze(mode)

    def _apply_freeze(self, mode: FreezeMode) -> None:
        if mode == "none":
            for p in self.vqgan.parameters():
                p.requires_grad = True
            self.vqgan.train()
            return
        if mode == "full":
            for p in self.vqgan.parameters():
                p.requires_grad = False
            self.vqgan.eval()
            return
        if mode == "partial":
            trainable_patterns = set(self._PARTIAL_FREEZE_TRAINABLE_PATTERNS)
            actual_names = {n for n, _ in self.vqgan.named_parameters()}
            # Only enable patterns that actually exist in this VQGAN
            # topology — keeps tiny / smoke configs working when they have
            # fewer than 3 decoder ResBlocks or an Identity post_quant.
            self._partial_trainable: set[str] = trainable_patterns & actual_names
            any_trainable = False
            for name, p in self.vqgan.named_parameters():
                if name in self._partial_trainable:
                    p.requires_grad = True
                    any_trainable = True
                else:
                    p.requires_grad = False
            if not any_trainable:
                raise RuntimeError(
                    "VQFont partial freeze matched zero parameters in the "
                    "current VQGAN topology. Expected at least one of: "
                    f"{sorted(trainable_patterns)}. Actual names start with: "
                    f"{sorted(list(actual_names))[:6]} ..."
                )
            self.vqgan.eval()  # InstanceNorm doesn't track stats so this is safe
            return
        raise ValueError(f"Unknown freeze_vqgan mode: {mode!r}")

    def _freeze_vqgan(self) -> None:
        """Legacy alias: full freeze (kept for backward compatibility)."""
        self._apply_freeze("full")

    def _vqgan_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run VQGAN encoder + pre_quant projection.

        Returns the continuous pre-quantization features. ``no_grad`` is
        applied when VQGAN is in ``full`` freeze mode; under ``partial`` we
        still need grad to flow into the trainable encoder params, even
        though by default the encoder is frozen — keeping the path
        differentiable is safer.
        """
        if self.freeze_mode == "full":
            with torch.no_grad():
                return self.vqgan.pre_quant(self.vqgan.encoder(x))
        return self.vqgan.pre_quant(self.vqgan.encoder(x))

    @torch.no_grad()
    def encode_target_indices(self, target_image: torch.Tensor) -> torch.Tensor:
        """Encode target glyph to [B, H_lat, W_lat] codebook indices."""
        return self.vqgan.encode_indices(target_image)

    def predict_token_logits(
        self,
        initial_glyph: torch.Tensor,
        ref_glyphs: torch.Tensor,
        structure_id: torch.Tensor,
        ref_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce per-token codebook logits + the post-SSEM attention map.

        Args:
            initial_glyph: [B, C, H, W] initial synthesized glyph (from any
                upstream FFG synthesis module). Phase 1: we accept the
                source/content image as a stand-in if no synthesis module
                checkpoint is loaded.
            ref_glyphs:    [B, R, C, H, W] R reference glyphs.
            structure_id:  [B] long structure class id (0..N-1).
            ref_valid:     [B, R] bool optional mask.

        Returns:
            (token_logits [B, N, K], attn_map [B, heads, N, R*N]).
        """
        b, r, c, h, w = ref_glyphs.shape
        q_feat = self._vqgan_encode(initial_glyph)
        ref_flat = ref_glyphs.reshape(b * r, c, h, w)
        ref_feat = self._vqgan_encode(ref_flat)
        # ref_feat: [B*R, C', H_lat, W_lat] -> [B, R, C', H_lat, W_lat]
        ref_feat = ref_feat.reshape(b, r, *ref_feat.shape[1:])
        return self.transformer(q_feat, ref_feat, structure_id, ref_valid=ref_valid)

    def decode_indices_to_image(self, indices: torch.Tensor) -> torch.Tensor:
        """Convenience: predicted indices -> reconstructed image."""
        return self.vqgan.decode_indices(indices)

    def forward(
        self,
        initial_glyph: torch.Tensor,
        ref_glyphs: torch.Tensor,
        structure_id: torch.Tensor,
        ref_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper around ``predict_token_logits``."""
        return self.predict_token_logits(initial_glyph, ref_glyphs, structure_id, ref_valid=ref_valid)


def build_vq_font(cfg: VQFontConfig, *, freeze_vqgan: FreezeMode | bool = "partial") -> VQFont:
    return VQFont(cfg, freeze_vqgan=freeze_vqgan)
