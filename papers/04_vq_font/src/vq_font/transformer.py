"""Token Prior Refinement Transformer + SSEM — Stage 1 of VQ-Font.

The paper note ([036_VQ-Font結構感知字體_AAAI2023]) describes a Transformer
that:
  * Takes an **initial synthesized glyph** (from any FFG synthesis module) plus
    **3 reference glyphs** and predicts codebook indices over the pre-trained
    VQGAN codebook (size 1024, 16x16 features) [paper-cited Phase 0 row 04].
  * Uses cross-attention (8 heads in the original config) against the 3
    references' encoder features — the references provide the style prior.
  * Is paired with the **Structure-level Style Enhancement Module (SSEM)**
    which projects the 12 Chinese-character structure classes (+ atomic) into
    a structure embedding that **conditions** the transformer; the head can
    also predict the structure id from the synthesized glyph as an auxiliary
    classification loss.

This module is intentionally token-centric: it operates on the encoder output
of a *frozen* VQGAN (loaded from Stage 0). The codebook is **not** modified.

`StructureEncoder` maps the 13-way structure id (12 Chinese structures from
`scripts/lookup_ids.py::parse_structure` + atomic fallback) to a learnable
embedding that is summed into the transformer input tokens. `StructureHead`
is a tiny classification head used by `compute_loss` for L_struct.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TransformerConfig",
    "StructureEncoder",
    "StructureHead",
    "TokenPriorTransformer",
    "build_transformer",
    "NUM_STRUCTURE_CLASSES",
]


# `lookup_ids.parse_structure` returns one of 12 structure names + 'atomic'.
# We allocate 13 classes (0..12) plus 1 sentinel for 'unknown' (when no IDS
# data is available) -> 14 classes total. Keeping 'unknown' explicit lets
# Stage A (TTF pretrain, no IDS lookup) cleanly route through the same head.
NUM_STRUCTURE_CLASSES: int = 14


@dataclass
class TransformerConfig:
    """Hyperparameters for the Token Prior Refinement Transformer."""

    image_size: int = 128
    """Pixel size; transformer operates in feature space H/8 x W/8."""
    latent_resolution: int = 16
    """Spatial size after VQGAN encoder. Should match VQGANConfig.out_resolution."""
    embed_dim: int = 256
    """Token width, matches VQGAN codebook embed_dim so we can read the
    encoder feature map directly as a token sequence."""
    num_blocks: int = 6
    """Transformer depth. Paper says "Transformer cross-attention 8 heads"
    but does not pin depth; 6 is a `[guessed-because-paper-vague]` default
    inherited from small VQ-VAE-2 / DALL-E priors."""
    num_heads: int = 8
    """Paper-cited 8 heads."""
    mlp_ratio: float = 4.0
    """Standard Transformer MLP expansion ratio."""
    dropout: float = 0.0
    num_refs: int = 3
    """Paper-cited 3 reference characters."""
    codebook_size: int = 1024
    """Output vocabulary; matches VQGANConfig.num_embeddings."""
    num_structures: int = NUM_STRUCTURE_CLASSES
    """12 Chinese structures + atomic + unknown sentinel = 14."""


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _build_pos_embed(n_positions: int, dim: int) -> nn.Parameter:
    """Learnable positional embedding initialized small."""
    pe = nn.Parameter(torch.zeros(1, n_positions, dim))
    nn.init.normal_(pe, std=0.02)
    return pe


# --------------------------------------------------------------------------------------
# Transformer blocks
# --------------------------------------------------------------------------------------


class _MultiHeadAttention(nn.Module):
    """Generic multi-head attention with optional separate K/V source."""

    def __init__(self, dim: int, num_heads: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"_MultiHeadAttention: dim={dim} must be divisible by num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kv = q if kv is None else kv
        b, n_q, c = q.shape
        n_kv = kv.shape[1]
        qh = self.q_proj(q).reshape(b, n_q, self.num_heads, self.head_dim).transpose(1, 2)
        kh = self.k_proj(kv).reshape(b, n_kv, self.num_heads, self.head_dim).transpose(1, 2)
        vh = self.v_proj(kv).reshape(b, n_kv, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(qh, kh.transpose(-1, -2)) * self.scale
        if mask is not None:
            # mask: [B, n_kv]; True = keep, False = mask out.
            attn = attn.masked_fill(~mask.view(b, 1, 1, n_kv), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, vh).transpose(1, 2).reshape(b, n_q, c)
        return self.out_proj(out)


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class _Block(nn.Module):
    """Transformer block with self-attn + cross-attn + MLP (Pre-LN)."""

    def __init__(self, dim: int, num_heads: int, *, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.self_attn = _MultiHeadAttention(dim, num_heads, dropout=dropout)
        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = _MultiHeadAttention(dim, num_heads, dropout=dropout)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm_q1(x))
        x = x + self.cross_attn(self.norm_q2(x), self.norm_kv(context), mask=context_mask)
        x = x + self.mlp(self.norm_mlp(x))
        return x


# --------------------------------------------------------------------------------------
# SSEM components
# --------------------------------------------------------------------------------------


class StructureEncoder(nn.Module):
    """Embed the 12-way Chinese structure class (+ atomic + unknown).

    The structure embedding is summed into every query token via FiLM-style
    additive bias (broadcast over the sequence). This is the lightweight form
    of SSEM conditioning; the paper describes SSEM as an enhancement that
    "matches style features at the structure level", so we additionally expose
    a structure-conditioned context vector that the transformer's cross-attn
    can attend to as the first prefix token of the reference context.
    """

    def __init__(self, *, num_structures: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_structures, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, structure_id: torch.Tensor) -> torch.Tensor:
        """structure_id: [B] long -> embedding [B, embed_dim]."""
        return self.embedding(structure_id)


class StructureHead(nn.Module):
    """Auxiliary head predicting the structure class from token features.

    Implements the SSEM "structure-level style matching" objective as a
    classification problem: we pool the transformer's hidden state and
    require it to be predictive of the target's structure id. This gives the
    cross-attention an additional gradient signal aligned with the 12
    Chinese-structure prior. `[guessed-because-paper-vague]` — paper says
    SSEM uses "structure-level matching"; we operationalise it as a CE loss
    over the 14-way structure id.
    """

    def __init__(self, *, embed_dim: int, num_structures: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, num_structures)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, N, C] -> logits [B, num_structures]."""
        pooled = self.norm(tokens).mean(dim=1)
        return self.proj(pooled)


# --------------------------------------------------------------------------------------
# Token Prior Refinement Transformer
# --------------------------------------------------------------------------------------


class TokenPriorTransformer(nn.Module):
    """Predict codebook indices for the target glyph from initial synthesis + refs.

    Pipeline (see `forward`):
      1. `query_feat` = VQGAN_enc(initial_synth) — [B, C, H, W], then flattened
         to a [B, N=H*W, C] sequence + positional embedding.
      2. `ref_feat`   = VQGAN_enc(refs)          — [B, R, C, H, W], flattened
         and concatenated along the sequence axis to [B, R*N, C] + positional
         embedding. The SSEM structure embedding is concatenated as a global
         prefix token (length 1) so cross-attention can route structure-aware
         gating.
      3. Stack of `num_blocks` transformer blocks consume query tokens with
         cross-attention over `ref_feat`.
      4. A linear head emits per-token logits over the codebook (size 1024).
      5. A SSEM `StructureHead` reads the final hidden state to produce the
         auxiliary structure classification.

    Notes:
      * The synthesis-module's output is approximated in Phase 1 by re-using
        VQGAN.encoder on the source content image — this is a `[guessed]`
        approximation (the paper's "any FFG synthesis module" is intentionally
        underspecified). For real Stage B/C training we expect an external
        synthesis module checkpoint to be loaded via train.yaml.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        n_query = cfg.latent_resolution * cfg.latent_resolution
        n_per_ref = n_query
        n_ref_total = cfg.num_refs * n_per_ref

        self.query_pos = _build_pos_embed(n_query, cfg.embed_dim)
        self.ref_pos = _build_pos_embed(n_ref_total + 1, cfg.embed_dim)
        # +1 for SSEM prefix token.

        self.struct_encoder = StructureEncoder(
            num_structures=cfg.num_structures, embed_dim=cfg.embed_dim
        )

        self.input_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.ref_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

        self.blocks = nn.ModuleList(
            [
                _Block(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_blocks)
            ]
        )

        self.out_norm = nn.LayerNorm(cfg.embed_dim)
        self.token_head = nn.Linear(cfg.embed_dim, cfg.codebook_size)
        self.structure_head = StructureHead(
            embed_dim=cfg.embed_dim, num_structures=cfg.num_structures
        )
        # Sentinel pad token for empty ref slots.
        self.ref_null_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        nn.init.normal_(self.ref_null_token, std=0.02)

    # --- helpers ---

    def _featmap_to_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, H*W, C]."""
        b, c, h, w = feat.shape
        return feat.flatten(2).transpose(1, 2)

    def _stack_refs(
        self,
        ref_feats: torch.Tensor,
        ref_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack [B, R, C, H, W] ref features into [B, R*H*W, C] + valid mask.

        ref_valid: [B, R] bool. When False at slot r, that ref's tokens are
        substituted by the learned null token so cross-attn still has finite
        keys (a fully-masked head would NaN under softmax).
        """
        b, r, c, h, w = ref_feats.shape
        tokens = ref_feats.flatten(3).permute(0, 1, 3, 2).reshape(b, r * h * w, c)
        if ref_valid is None:
            mask = torch.ones(b, r * h * w, dtype=torch.bool, device=tokens.device)
            return tokens, mask
        per_ref = h * w
        valid_expanded = ref_valid.unsqueeze(-1).expand(-1, -1, per_ref).reshape(b, r * per_ref)
        null = self.ref_null_token.expand(b, r * per_ref, c)
        tokens = torch.where(valid_expanded.unsqueeze(-1), tokens, null)
        return tokens, valid_expanded

    # --- main ---

    def forward(
        self,
        query_feat: torch.Tensor,
        ref_feats: torch.Tensor,
        structure_id: torch.Tensor,
        ref_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run token-prior refinement.

        Args:
            query_feat:  [B, embed_dim, H_lat, W_lat] — VQGAN-encoder feature
                of the initial synthesized glyph (continuous, pre-quantization).
            ref_feats:   [B, R, embed_dim, H_lat, W_lat] — encoder features of
                the R reference glyphs.
            structure_id: [B] long — Chinese-structure class id (0..N_struct-1)
                from `lookup_ids.parse_structure` mapping.
            ref_valid:   [B, R] bool or None — which ref slots are real.

        Returns:
            token_logits: [B, H_lat*W_lat, codebook_size]
            structure_logits: [B, num_structures] from the SSEM aux head.
        """
        q_tokens = self.input_proj(self._featmap_to_tokens(query_feat))
        q_tokens = q_tokens + self.query_pos

        # SSEM additive structure bias on the queries.
        struct_emb = self.struct_encoder(structure_id)               # [B, C]
        q_tokens = q_tokens + struct_emb.unsqueeze(1)

        # References + SSEM prefix token in the context sequence.
        ref_tokens, ref_mask = self._stack_refs(ref_feats, ref_valid)
        ref_tokens = self.ref_proj(ref_tokens)
        struct_prefix = struct_emb.unsqueeze(1)                      # [B, 1, C]
        ref_tokens = torch.cat([struct_prefix, ref_tokens], dim=1)
        ref_tokens = ref_tokens + self.ref_pos
        # Extend mask with always-valid SSEM prefix.
        prefix_valid = torch.ones(
            ref_mask.shape[0], 1, dtype=torch.bool, device=ref_mask.device
        )
        ref_mask = torch.cat([prefix_valid, ref_mask], dim=1)

        h = q_tokens
        for block in self.blocks:
            h = block(h, ref_tokens, context_mask=ref_mask)

        h = self.out_norm(h)
        token_logits = self.token_head(h)
        structure_logits = self.structure_head(h)
        return token_logits, structure_logits


def build_transformer(cfg: TransformerConfig) -> TokenPriorTransformer:
    return TokenPriorTransformer(cfg)
