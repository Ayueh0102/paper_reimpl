"""Top-level VQ-Font composite model.

VQ-Font is a two-stage system: Stage 0 trains a VQGAN font codebook end-to-end
(`vqgan.VQGAN`), and Stage 1+ trains a Transformer that refines token indices
against the frozen codebook (`transformer.TokenPriorTransformer`).

This module exposes both pieces in one place so train.py / sample.py / tests
can do a single ``from vq_font.model import build_vq_font``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .transformer import (
    NUM_STRUCTURE_CLASSES,
    StructureEncoder,
    StructureHead,
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
    "StructureEncoder",
    "StructureHead",
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
        # Catch latent-grid mismatches at config-build time so a forward-pass
        # shape error doesn't surprise us mid-training.
        vqgan_latent = self.vqgan.out_resolution()
        if self.transformer.latent_resolution != vqgan_latent:
            raise ValueError(
                "VQFontConfig: transformer.latent_resolution must equal "
                f"vqgan.out_resolution(); got {self.transformer.latent_resolution} "
                f"vs {vqgan_latent} (vqgan.image_size={self.vqgan.image_size}, "
                f"channel_mult={self.vqgan.channel_mult})"
            )


class VQFont(nn.Module):
    """Stage-1+ composite: frozen VQGAN + trainable Transformer.

    The VQGAN can be wired as ``frozen=True`` (Stage 1+ paper recipe) or
    ``frozen=False`` (Stage 0 still trainable). The Transformer is always
    trainable. The class deliberately keeps both pieces exposed as attributes
    so external code can do parameter-group splits (e.g. AdamW different lrs).
    """

    def __init__(self, cfg: VQFontConfig, *, freeze_vqgan: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.vqgan = build_vqgan(cfg.vqgan)
        self.transformer = build_transformer(cfg.transformer)
        if freeze_vqgan:
            self._freeze_vqgan()

    def _freeze_vqgan(self) -> None:
        for p in self.vqgan.parameters():
            p.requires_grad = False
        self.vqgan.eval()

    @torch.no_grad()
    def _vqgan_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run VQGAN encoder + pre_quant projection, returning continuous features."""
        z_e = self.vqgan.pre_quant(self.vqgan.encoder(x))
        return z_e

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
        """Produce per-token codebook logits + SSEM structure logits.

        Args:
            initial_glyph: [B, C, H, W] initial synthesized glyph (from any
                upstream FFG synthesis module). Phase 1: we accept the
                source/content image as a stand-in if no synthesis module
                checkpoint is loaded.
            ref_glyphs:    [B, R, C, H, W] R reference glyphs.
            structure_id:  [B] long ids into the 14-way structure vocab.
            ref_valid:     [B, R] bool optional mask.

        Returns:
            (token_logits [B, N, K], structure_logits [B, num_structures]).
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
        """Convenience wrapper around `predict_token_logits`."""
        return self.predict_token_logits(initial_glyph, ref_glyphs, structure_id, ref_valid=ref_valid)


def build_vq_font(cfg: VQFontConfig, *, freeze_vqgan: bool = True) -> VQFont:
    return VQFont(cfg, freeze_vqgan=freeze_vqgan)
