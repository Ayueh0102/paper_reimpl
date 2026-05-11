"""Token Prior Refinement Transformer + SSEM — Stage 1 of VQ-Font.

Phase 2 (post-github-diff) note
-------------------------------
Two architectural decisions changed materially against the blind-impl:

1. **Stack**: was 6 blocks of self-attn + cross-attn + MLP. Official
   (``models/generator.py:51-52`` and ``models/former.py:16-51``) is
   **15 self-attention-only blocks**; the content↔style cross-attention
   is computed *before* the stack via explicit Linear K/Q/V projections
   on the 16x16 content feature map vs the ``R*16*16`` reference tokens
   (``generator.py:read_decode`` 962-994).

2. **SSEM**: was a learned ``StructureEncoder`` additive bias + an
   auxiliary ``StructureHead`` cross-entropy loss. Official SSEM is
   parameter-free — it is a **hand-coded per-class spatial recalibration
   of the cross-attention map** before the softmax (``generator.py:929``
   — ``fusion_atten``). ``cont_similarity`` / ``refer_similarity`` /
   ``fusion_am`` partition the 16x16 attention map according to the 13
   structure classes (top-bottom, left-mid-right, surrounds, etc.) and
   average each region; the regional averages are then added back to the
   per-token logits before softmax. Class 1, 3, 5, 6, 7, 11, 12 simply
   take the full map (single-region), while classes 0, 4, 8, 9 split into
   2 regions, classes 2 and 10 into 3 regions. The exact slice ranges are
   reproduced here in ``REGION_TEMPLATES``.

This module now exposes:

* ``REGION_TEMPLATES`` — the 13 ``(name, [(slice_h, slice_w), ...])``
  templates lifted from ``generator.py:cont_similarity`` (lines 129-211).
  Each template's region count equals the official ``num`` returned by
  ``refer_similarity``.
* ``RegionAttentionRecalibrator`` — applies the region-pooled bias to a
  ``[B, heads, N_q, N_kv]`` cross-attn map prior to softmax. Pure-tensor,
  no learnable parameters (matches the paper-faithful form).
* ``TokenPriorTransformer`` — drives the full pipeline: linear K/Q/V →
  attn map → SSEM recalibration → softmax → matmul V → LayerNorm → 15
  self-attn blocks → ``mlp_head`` over the 1024-entry codebook.

Class count: **13** (0..12) per ``meta/stru_all.json``. The 14th
"unknown" sentinel in the blind-impl is dropped; manifests that lack a
structure id should pre-compute one via ``meta/stru_all.json`` or
``lookup_ids.parse_structure`` at build time. The legacy 14-way head
remains supported only for compatibility with old checkpoints (passing
``num_structures=14`` still works but maps the extra classes to the
nearest official template; see ``RegionAttentionRecalibrator``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TransformerConfig",
    "RegionAttentionRecalibrator",
    "TokenPriorTransformer",
    "build_transformer",
    "NUM_STRUCTURE_CLASSES",
    "REGION_TEMPLATES",
]


# Official vocabulary: 13 classes 0..12 per ``meta/stru_all.json``.
NUM_STRUCTURE_CLASSES: int = 13


# --------------------------------------------------------------------------------------
# Region templates lifted from ``models/generator.py:cont_similarity`` (129-211)
# --------------------------------------------------------------------------------------
#
# Each entry maps a structure id to a list of (h_slice, w_slice) pairs that
# partition the 16x16 content feature map. Classes with a single full-map
# region (1, 3, 5, 6, 7, 11, 12) end up as a no-op identity bias (the same
# region average added uniformly to all tokens) — kept for symmetry with the
# official code path. Slice indices match the official line-level numbers
# (e.g. tar_stru==0: ``i_[:7, :]`` vs ``i_[7:, :]`` → top/bottom split at
# row 7, ``generator.py:134-137``).
#
# The official ``cont_similarity`` and ``refer_similarity`` always use the
# *same* spatial partition for a given structure id, so we encode it once.
# Composite classes (8, 9) blend two sub-regions with fixed weights
# (104/152 + 48/152 etc.); we expose those as (slice, weight) tuples on a
# second axis when needed — see ``REGION_BLENDS`` below.

# Type alias: a region = list[(h_slice, w_slice, weight)] — weights sum
# over the region's sub-blocks.
_FULL_H = slice(0, 16)
_FULL_W = slice(0, 16)

REGION_TEMPLATES: dict[int, list[list[tuple[slice, slice, float]]]] = {
    0: [
        # Top-bottom split at row 7 (``i_[:7, :]`` vs ``i_[7:, :]``).
        [(slice(0, 7), _FULL_W, 1.0)],
        [(slice(7, 16), _FULL_W, 1.0)],
    ],
    1: [
        # Single region (full map).
        [(_FULL_H, _FULL_W, 1.0)],
    ],
    2: [
        # Three rows: 0:5 / 5:8 / 8:.
        [(slice(0, 5), _FULL_W, 1.0)],
        [(slice(8, 16), _FULL_W, 1.0)],
        [(slice(5, 8), _FULL_W, 1.0)],
    ],
    3: [
        [(_FULL_H, _FULL_W, 1.0)],
    ],
    4: [
        # Left-right split at column 7.
        [(_FULL_H, slice(0, 7), 1.0)],
        [(_FULL_H, slice(7, 16), 1.0)],
    ],
    5: [
        [(_FULL_H, _FULL_W, 1.0)],
    ],
    6: [
        [(_FULL_H, _FULL_W, 1.0)],
    ],
    7: [
        [(_FULL_H, _FULL_W, 1.0)],
    ],
    8: [
        # Composite class with a fixed-weight blend:
        #   region a = i_[:-3, 8:-3]
        #   region b = i_[:-3, :8] * (104/152) + i_[-3:, :] * (48/152)
        [(slice(0, 13), slice(8, 13), 1.0)],
        [
            (slice(0, 13), slice(0, 8), 104.0 / 152.0),
            (slice(13, 16), _FULL_W, 48.0 / 152.0),
        ],
    ],
    9: [
        # Composite blend:
        #   region a = i_[6:, 5:]
        #   region b = i_[:6, :] * (96/146) + i_[6:, :5] * (50/146)
        [(slice(6, 16), slice(5, 16), 1.0)],
        [
            (slice(0, 6), _FULL_W, 96.0 / 146.0),
            (slice(6, 16), slice(0, 5), 50.0 / 146.0),
        ],
    ],
    10: [
        # Three columns: 0:6 / 7:11 / 11:.
        [(_FULL_H, slice(0, 6), 1.0)],
        [(_FULL_H, slice(7, 11), 1.0)],
        [(_FULL_H, slice(11, 16), 1.0)],
    ],
    11: [
        [(_FULL_H, _FULL_W, 1.0)],
    ],
    12: [
        [(_FULL_H, _FULL_W, 1.0)],
    ],
}

assert len(REGION_TEMPLATES) == NUM_STRUCTURE_CLASSES, (
    f"REGION_TEMPLATES has {len(REGION_TEMPLATES)} entries but "
    f"NUM_STRUCTURE_CLASSES={NUM_STRUCTURE_CLASSES}"
)


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
    num_blocks: int = 15
    """Self-attention depth. Official: 15 blocks (``generator.py:51``)."""
    num_heads: int = 8
    """Paper-cited 8 heads (``generator.py:51``, ``cfgs/custom.yaml:12``)."""
    mlp_ratio: float = 2.0
    """Self-attn block MLP expansion ratio. Official ``dim_mlp=512`` with
    ``embed_dim=256`` -> ratio=2 (``former.py:17`` default 2048 is overridden
    in ``generator.py:51`` to 512)."""
    dropout: float = 0.0
    num_refs: int = 3
    """Paper-cited 3 reference characters (``cfgs/custom.yaml:13`` kshot)."""
    codebook_size: int = 1024
    """Output vocabulary; matches VQGANConfig.num_embeddings."""
    num_structures: int = NUM_STRUCTURE_CLASSES
    """13 official classes from ``meta/stru_all.json``."""


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _build_pos_embed(n_positions: int, dim: int) -> nn.Parameter:
    """Learnable positional embedding initialized small.

    Official ``generator.py:53`` initializes ``position_emb = zeros(256, 256)``
    (i.e. ``[N=H*W, C]``) and adds it as ``query_pos`` inside each
    ``TransformerSALayer`` (``former.py:42-44``). We keep ours zero-init too
    so behavior matches at step 0.
    """
    pe = nn.Parameter(torch.zeros(1, n_positions, dim))
    return pe


# --------------------------------------------------------------------------------------
# SSEM — per-class spatial recalibration of the cross-attention map
# --------------------------------------------------------------------------------------


class RegionAttentionRecalibrator(nn.Module):
    """Pure-tensor SSEM: add region-pooled bias to cross-attn logits.

    Reproduces ``fusion_atten`` + ``cont_similarity`` + ``refer_similarity``
    + ``fusion_am`` from ``models/generator.py``. For each sample in the
    batch we:

    1. Reshape the attention logits to ``[heads, H, W, R, H, W]``
       (queries: content tokens at HxW; keys: R reference HxW grids).
    2. For each query-region defined by the target ``structure_id`` and
       each key-region defined by the reference ``structure_id`` (same
       id for both queries and the matching reference per the official
       behaviour when ``trg_stru_ids.shape == in_stru_ids.shape``), compute
       the mean attention logit inside that ``[query-region × key-region]``
       block.
    3. Add that mean back into each per-token logit within the same
       ``[query-region × key-region]`` block, then return the recalibrated
       logits for downstream softmax.

    No learnable parameters. Matches the official approach.
    """

    def __init__(self, *, latent_resolution: int = 16,
                 num_structures: int = NUM_STRUCTURE_CLASSES) -> None:
        super().__init__()
        self.latent_resolution = int(latent_resolution)
        self.num_structures = int(num_structures)
        if self.latent_resolution != 16:
            # The official template slices were designed for a 16x16 grid
            # (``meta/stru_all.json`` data is 16x16-only). Other sizes will
            # use the template's nearest-fraction interpretation below.
            pass  # We support generic latents by scaling slices in forward().

    def _scaled_slice(self, s: slice, dim: int) -> slice:
        """Scale a 16-grid slice down to ``dim`` linearly, rounding to int.

        For the official 16x16 grid this is a no-op. For smoke-test 4x4
        grids it scales ``slice(0, 7)`` → ``slice(0, round(7/16*4))``. We
        clamp to ``[0, dim]`` and guarantee ``start < stop`` (otherwise the
        region degenerates and contributes 0 mean, which is what the
        official code does anyway when called on a tiny latent).
        """
        if dim == 16:
            return s
        start = int(round(s.start * dim / 16.0))
        stop = int(round(s.stop * dim / 16.0))
        start = max(0, min(dim, start))
        stop = max(start + 1, min(dim, stop))
        return slice(start, stop)

    def forward(
        self,
        attn_logits: torch.Tensor,
        structure_id: torch.Tensor,
        *,
        num_refs: int,
    ) -> torch.Tensor:
        """Apply region-pooled bias to pre-softmax cross-attn logits.

        Args:
            attn_logits: ``[B, heads, H*W, R*H*W]`` — content-to-references
                cross-attention logits, pre-softmax.
            structure_id: ``[B]`` long — target structure class (0..N-1).
                Per the official behaviour with ``trg_stru_ids.shape ==
                in_stru_ids.shape``, the same id is used for both query and
                key partitioning.
            num_refs: int — R, the number of reference characters per sample.

        Returns:
            Recalibrated logits, same shape as ``attn_logits``.
        """
        b, heads, n_q, n_kv = attn_logits.shape
        h_w = int(round(math.sqrt(n_q)))
        assert h_w * h_w == n_q, (
            f"attn_logits: n_q={n_q} is not a perfect square; "
            f"RegionAttentionRecalibrator expects a square HxW content grid."
        )
        assert n_kv == num_refs * n_q, (
            f"attn_logits: n_kv={n_kv} != num_refs*n_q ({num_refs}*{n_q})"
        )

        # Reshape: ``[B, heads, H_q, W_q, R, H_kv, W_kv]``. We work on a
        # view of the original logits so the additive bias is naturally
        # consistent with the source layout. The view shares storage; we
        # use a fresh out-of-place ``+`` at the end so autograd handles it.
        a = attn_logits.reshape(b, heads, h_w, h_w, num_refs, h_w, h_w)

        # Build the additive bias to inject. Same shape as ``a`` (7-D).
        bias = torch.zeros_like(a)

        for i in range(b):
            sid = int(structure_id[i].item())
            sid = max(0, min(self.num_structures - 1, sid))
            # Out-of-range sid (e.g. legacy 13 = "overlap" or 14 = "unknown")
            # is clamped to the last available template. Default templates
            # 11/12 are "atomic/single" full-map regions, which is the
            # paper-faithful fallback.
            template = REGION_TEMPLATES.get(sid, REGION_TEMPLATES[NUM_STRUCTURE_CLASSES - 1])
            for q_region in template:
                for r in range(num_refs):
                    for k_region in template:
                        # ``block``: ``[heads, H_q, W_q, H_kv, W_kv]``.
                        block = a[i, :, :, :, r, :, :]
                        # Mean over the [Q-region × K-region] block.
                        block_mean = self._region_mean_qk(block, q_region, k_region, h_w)
                        # Scatter mean back into the same [Q-region × K-region] block.
                        self._scatter_qk(
                            bias[i, :, :, :, r, :, :],
                            block_mean,
                            q_region, k_region, h_w,
                        )

        # ``a`` and ``bias`` are 7-D views; we need to broadcast back to
        # ``attn_logits``'s 4-D shape ``[B, heads, H*W, R*H*W]``.
        bias_4d = bias.reshape(b, heads, n_q, n_kv)
        return attn_logits + bias_4d

    def _region_mean_qk(
        self,
        block: torch.Tensor,
        q_region: list[tuple[slice, slice, float]],
        k_region: list[tuple[slice, slice, float]],
        dim: int,
    ) -> torch.Tensor:
        """Weighted mean of ``block[..., Q-region, K-region]``.

        ``block``: ``[heads, H_q, W_q, H_kv, W_kv]``. Sub-blocks defined by
        Q-region and K-region are gathered and averaged. Returns
        ``[heads]`` (block prefix without the four spatial dims).

        For single-region templates (weight=1.0): unweighted mean over the
        concatenation of all sub-blocks (matches the official
        ``torch.mean(i_[..., slice, slice])`` form).

        For composite templates (classes 8 and 9 with fractional weights):
        weight each sub-block's mean by its declared weight and sum, which
        matches the official ``(104/152) * mean(A) + (48/152) * mean(B)``
        pattern.
        """
        pieces: list[torch.Tensor] = []
        weights: list[float] = []
        for qh, qw, q_weight in q_region:
            qh2 = self._scaled_slice(qh, dim)
            qw2 = self._scaled_slice(qw, dim)
            for kh, kw, k_weight in k_region:
                kh2 = self._scaled_slice(kh, dim)
                kw2 = self._scaled_slice(kw, dim)
                # Multi-axis slicing works in a single subscript when each
                # axis uses a slice (no fancy advanced indexing). ``block``
                # has 5 dims: [heads, H_q, W_q, H_kv, W_kv].
                sub = block[:, qh2, qw2, kh2, kw2]
                # ``sub``: [heads, h_q, w_q, h_kv, w_kv]
                pieces.append(sub.reshape(sub.shape[0], -1))
                weights.append(q_weight * k_weight)
        if not pieces:
            return block.new_zeros(block.shape[0])
        if all(w == 1.0 for w in weights):
            cat = torch.cat(pieces, dim=-1)
            return cat.mean(dim=-1)
        weighted_sum: torch.Tensor | None = None
        for piece, w in zip(pieces, weights, strict=True):
            piece_mean = piece.mean(dim=-1) * w
            weighted_sum = piece_mean if weighted_sum is None else weighted_sum + piece_mean
        assert weighted_sum is not None
        return weighted_sum

    def _scatter_qk(
        self,
        bias: torch.Tensor,
        value: torch.Tensor,
        q_region: list[tuple[slice, slice, float]],
        k_region: list[tuple[slice, slice, float]],
        dim: int,
    ) -> None:
        """Add ``value`` (broadcast) across the Q-region × K-region of ``bias``.

        ``bias``: ``[heads, H_q, W_q, H_kv, W_kv]`` (in-place add).
        ``value``: ``[heads]`` — per-head scalar to scatter. Sub-block weights
        scale how much of ``value`` lands in each sub-block.
        """
        for qh, qw, q_weight in q_region:
            qh2 = self._scaled_slice(qh, dim)
            qw2 = self._scaled_slice(qw, dim)
            for kh, kw, k_weight in k_region:
                kh2 = self._scaled_slice(kh, dim)
                kw2 = self._scaled_slice(kw, dim)
                w = q_weight * k_weight
                broadcast = value[:, None, None, None, None] * w
                # Use a single multi-axis subscript assignment — works on
                # contiguous slice indexing.
                bias[:, qh2, qw2, kh2, kw2] = bias[:, qh2, qw2, kh2, kw2] + broadcast


# --------------------------------------------------------------------------------------
# Self-attention only block (matches ``models/former.py:TransformerSALayer``)
# --------------------------------------------------------------------------------------


class _SelfAttnBlock(nn.Module):
    """Pre-LN self-attention + MLP block. No cross-attn inside.

    Matches ``former.py:TransformerSALayer``: ``LN → MHSA → +res →
    LN → Linear → GELU → Linear → +res`` with ``query_pos`` added to
    queries/keys (relative positional embedding pattern).
    """

    def __init__(self, dim: int, num_heads: int, *, dim_mlp: int, dropout: float = 0.0) -> None:
        super().__init__()
        # Use the same nn.MultiheadAttention as the official code so the
        # numerics match exactly (modulo init).
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim_mlp)
        self.linear2 = nn.Linear(dim_mlp, dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    @staticmethod
    def _with_pos(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(self, tgt: torch.Tensor, query_pos: torch.Tensor | None = None) -> torch.Tensor:
        """``tgt``: ``[N, B, C]`` — note: nn.MultiheadAttention default
        ``batch_first=False``. We follow the official format exactly
        (``former.py:35-50``).
        """
        tgt2 = self.norm1(tgt)
        q = k = self._with_pos(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, need_weights=False)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


# --------------------------------------------------------------------------------------
# Token Prior Refinement Transformer
# --------------------------------------------------------------------------------------


class TokenPriorTransformer(nn.Module):
    """Predict codebook indices for the target glyph from content + refs + SSEM.

    Pipeline (matches ``generator.py:read_decode`` 962-1002):

    1. ``content_feats = content_encoder(content_imgs)`` — done by caller
       (we receive the projected ``[B, C, H, W]`` query feature directly).
    2. Linear K/Q/V projections over flattened tokens.
    3. Multi-head scaled dot-product attention → ``[B, heads, H*W, R*H*W]``.
    4. **SSEM** ``RegionAttentionRecalibrator`` adds region-pooled bias.
    5. Softmax + matmul V → fused feature ``[B, H*W, C]`` → LayerNorm.
    6. Reshape to ``[B, C, H, W]``, flatten to ``[N, B, C]``, run 15
       self-attn blocks with shared learned ``query_pos``.
    7. ``mlp_head`` produces ``[B, N, codebook_size]`` logits.

    The caller is responsible for the front half (VQGAN encoder on the
    content image, encoder + memory write/read on the references). This
    class consumes already-encoded ``[B, C, H, W]`` content + ``[B, R, C, H, W]``
    references and returns logits + the recalibrated attention map (the
    latter for debug / aux losses).
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.embed_dim // cfg.num_heads
        if self.head_dim * cfg.num_heads != cfg.embed_dim:
            raise ValueError(
                f"embed_dim={cfg.embed_dim} must be divisible by num_heads={cfg.num_heads}"
            )

        # K/Q/V projections — bias=False matches official (``generator.py:66-68``).
        self.linears_key = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.linears_value = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.linears_query = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.layer_norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)

        # SSEM recalibrator — parameter-free.
        self.ssem = RegionAttentionRecalibrator(
            latent_resolution=cfg.latent_resolution,
            num_structures=cfg.num_structures,
        )

        dim_mlp = int(round(cfg.embed_dim * cfg.mlp_ratio))
        self.former = nn.ModuleList(
            [
                _SelfAttnBlock(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    dim_mlp=dim_mlp,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_blocks)
            ]
        )

        n_query = cfg.latent_resolution * cfg.latent_resolution
        # Official: ``nn.Parameter(torch.zeros(256, 256))`` — i.e. ``[N, C]``.
        # We keep the same shape so checkpoint keys align (``position_emb``).
        self.position_emb = nn.Parameter(torch.zeros(n_query, cfg.embed_dim))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.codebook_size),
        )

        # Sentinel pad token for empty ref slots (kept for API
        # compatibility with the blind impl smoke test; substituted into
        # an entire reference's tokens when ``ref_valid[i, r] = False``).
        self.ref_null_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        nn.init.normal_(self.ref_null_token, std=0.02)

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
            query_feat:  ``[B, embed_dim, H_lat, W_lat]`` — content image
                feature (the VQGAN encoder output for the source content).
            ref_feats:   ``[B, R, embed_dim, H_lat, W_lat]`` — reference
                feature maps.
            structure_id: ``[B]`` long — Chinese-structure class id (0..N-1).
            ref_valid:   ``[B, R]`` bool or None — which ref slots are real.

        Returns:
            token_logits: ``[B, H_lat*W_lat, codebook_size]``
            attn_map:     ``[B, num_heads, H_lat*W_lat, R*H_lat*W_lat]``
                (post-SSEM, pre-softmax — surfaced for ablations).
        """
        b, r, c, h, w = ref_feats.shape
        n_q = h * w
        if ref_valid is not None:
            # Replace masked-out refs with the learned null token before
            # cross-attention. This preserves finite logits everywhere.
            null = self.ref_null_token.expand(b * r, n_q, c).reshape(b, r, c, h, w)
            # ``ref_valid``: ``[B, R]`` -> broadcast to ``[B, R, 1, 1, 1]``.
            mask = ref_valid.view(b, r, 1, 1, 1)
            ref_feats = torch.where(mask, ref_feats, null)

        # ----- KQV projections + multi-head cross-attention -----
        # Content (query) tokens: ``[B, n_q, C]``.
        content_tokens = query_feat.flatten(2).transpose(1, 2)
        q = self.linears_query(content_tokens)
        # Reference tokens: ``[B, R*n_q, C]``.
        ref_tokens = ref_feats.permute(0, 1, 3, 4, 2).reshape(b, r * n_q, c)
        k = self.linears_key(ref_tokens)
        v = self.linears_value(ref_tokens)

        # Reshape for multi-head: ``[B, heads, N, head_dim]``.
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(t.shape[0], t.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        qh = split_heads(q)     # [B, heads, n_q, head_dim]
        kh = split_heads(k)     # [B, heads, R*n_q, head_dim]
        vh = split_heads(v)     # [B, heads, R*n_q, head_dim]

        # Scaled dot-product logits. Official scales by ``sqrt(h*w)`` (not
        # ``sqrt(head_dim)``) — see ``generator.py:982``. We mirror that.
        attn_logits = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(n_q)
        # [B, heads, n_q, R*n_q]

        # ----- SSEM (parameter-free recalibration) -----
        attn_logits = self.ssem(attn_logits, structure_id, num_refs=r)
        attn = F.softmax(attn_logits, dim=-1)

        # ----- Fuse -----
        fused = torch.matmul(attn, vh)  # [B, heads, n_q, head_dim]
        fused = fused.transpose(1, 2).reshape(b, n_q, c)
        fused = self.layer_norm(fused)

        # ----- 15 self-attn blocks (operating on ``[N, B, C]``) -----
        x = fused.transpose(0, 1)  # [N, B, C]
        # ``query_pos``: official ``self.position_emb.unsqueeze(1).repeat(1, B, 1)``.
        pos = self.position_emb.unsqueeze(1).expand(-1, b, -1)
        for blk in self.former:
            x = blk(x, query_pos=pos)
        x = x.transpose(0, 1)  # [B, N, C]

        # ----- Codebook logits -----
        token_logits = self.mlp_head(x)  # [B, N, K]
        return token_logits, attn_logits


def build_transformer(cfg: TransformerConfig) -> TokenPriorTransformer:
    return TokenPriorTransformer(cfg)
