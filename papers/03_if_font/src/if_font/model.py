"""IF-Font — Phase 2 corrected implementation.

Architecture (after Phase 2 alignment to official Stareven233/IF-Font):

  1. VQ tokenizer **frozen pretrained** CompVis ``vq-f8-n256`` (NOT trained
     from scratch). The blind Phase 1 trained its own VQGAN with embed_dim=256,
     grayscale, MSE-only loss — `reports/github_diff.md` items A2/D5/P0 #2 #7.
     We now expose a `VQTokenizerAdapter` that:
       * Defaults to RGB (in_channels=3), latent embed_dim=4, codebook 256,
         downsample 8 — the exact shape of the pretrained CompVis model.
       * Is **frozen** at construction: ``requires_grad_(False)`` + ``.eval()``.
       * For smoke / CI we instantiate a randomly-initialised tokenizer with
         the correct shapes; loading the real pretrained weights (via
         taming-transformers) is done outside this file by passing the
         already-frozen `nn.Module` in.
       * Provides ``encode(image)``, ``lookup(indices)``, ``decode_indices``.
     There is no Stage A VQGAN pretraining — VQ is fixed external state.

  2. **IDS encoder = bare `nn.Embedding` + a second `nn.Embedding`** (the
     dual-table design from official `encoder.py:191-192`). The previous
     full Transformer encoder is gone (over-engineered; A5 in github_diff).
     `embedding` feeds the AR prefix; `embedding2` feeds the 3SA cross-attn.

  3. **StyleEncoder + 3SA**: a CNN stem over each ref's quantised latent
     ([B*N, 4, 16, 16] → [B*N, c, 16, 16]), then a coverage-weighted average
     across the N refs (`x_g`) plus an IDS-conditioned cross-attention block
     (`_structure_style_aggregation`) that yields `x_l`. The final style
     sequence is `cat([x_l, x_g], dim=1)`.

  4. **MoCo wrapper**: two StyleEncoders (query + momentum-updated key),
     a 2-MLP projector + predictor head over `x_g`. Produces both `x_sss`
     (the style sequence for the AR decoder) and `cl` (the contrastive
     features for sup_cl loss). Official `encoder.MoCoWrapper`.

  5. **AR Transformer decoder**: 10 blocks · 8 heads · d_model 384. Each block
     = **1 self-attn + 1 cross-attn + 1 FFN** (NOT 2+1 as the paper note
     claimed). Pre-LN. Per-head QK-LayerNorm. The decoder consumes
     `x_sss` as cross-attn K/V and the IDS embedding as a **prefix prepended
     to the target token sequence** (per official `nanogpt.py:241`). Weight
     tying between `wte` and `lm_head`.

Conditioning summary (official-aligned):
  * IDS prefix-prepended to AR target embeddings; sliced off the logits tail
    so CE only contributes from the image-token positions.
  * Style sequence `x_sss = cat([x_l, x_g])` is the cross-attn K/V.
  * No CFG (the paper does not use it).

Losses (training):
  * `losses.sq` — cross-entropy on next VQ token (target = quantised target
    glyph indices).
  * `losses.sup_cl` — supervised contrastive on style features keyed by
    font_id / writer_id.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================
# VQ tokenizer adapter (frozen pretrained CompVis vq-f8-n256)
# ======================================================================


@dataclass
class VQTokenizerConfig:
    """Shape contract for the frozen VQGAN.

    Matches CompVis pretrained `vq-f8-n256`:
      * image_size = 128 (RGB)
      * in_channels = 3
      * embedding_dim = 4   (NOT 256 — that was the blind Phase 1 mistake)
      * codebook_size = 256
      * downsample factor = 8 → 16×16 = 256 tokens per glyph
    """

    image_size: int = 128
    in_channels: int = 3
    embedding_dim: int = 4
    codebook_size: int = 256
    downsample_factor: int = 8

    @property
    def token_grid_size(self) -> int:
        return self.image_size // self.downsample_factor

    @property
    def n_tokens(self) -> int:
        return self.token_grid_size ** 2


def _gn(channels: int) -> nn.GroupNorm:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


class _ResnetBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.norm1 = _gn(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = _gn(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class _StubVQGANEncoder(nn.Module):
    """3-downsample CNN producing a [B, embed_dim, H/8, W/8] latent.

    This is NOT the real pretrained CompVis model — it is a shape-matching
    stub used when running smoke tests or when the real pretrained weights
    are unavailable. The real model can be substituted via
    ``VQTokenizerAdapter.load_pretrained(vqgan_path)``.
    """

    def __init__(self, cfg: VQTokenizerConfig) -> None:
        super().__init__()
        ch = (32, 64, 64, 64)
        self.in_conv = nn.Conv2d(cfg.in_channels, ch[0], 3, padding=1)
        layers: list[nn.Module] = []
        prev = ch[0]
        for c in ch[1:]:
            layers.append(_ResnetBlock(prev, c))
            layers.append(nn.Conv2d(c, c, 3, stride=2, padding=1))  # downsample
            prev = c
        layers.append(_ResnetBlock(prev, prev))
        self.body = nn.Sequential(*layers)
        self.out_norm = _gn(prev)
        self.out_conv = nn.Conv2d(prev, cfg.embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.body(x)
        return self.out_conv(F.silu(self.out_norm(x)))


class _StubVQGANDecoder(nn.Module):
    """3-upsample CNN inverse of the stub encoder."""

    def __init__(self, cfg: VQTokenizerConfig) -> None:
        super().__init__()
        ch = (64, 64, 64, 32)
        self.in_conv = nn.Conv2d(cfg.embedding_dim, ch[0], 3, padding=1)
        layers: list[nn.Module] = []
        prev = ch[0]
        for c in ch[1:]:
            layers.append(_ResnetBlock(prev, c))
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            prev = c
        layers.append(_ResnetBlock(prev, prev))
        self.body = nn.Sequential(*layers)
        self.out_norm = _gn(prev)
        self.out_conv = nn.Conv2d(prev, cfg.in_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.in_conv(z)
        z = self.body(z)
        return self.out_conv(F.silu(self.out_norm(z)))


class VQTokenizerAdapter(nn.Module):
    """Frozen pretrained-VQGAN tokenizer wrapper.

    Public API (aligned with official `data.adapter.VQAdapter`):
      * ``encode(image) -> indices [B, n_tokens]`` — codebook indices.
      * ``lookup(indices) -> quantised [B, n_tokens, embed_dim]`` (flat) or
        ``lookup_quant(indices) -> [B, embed_dim, H, W]`` (grid).
      * ``decode_indices(indices) -> image [B, in_channels, H, W]``.
      * ``get_codebook() -> [codebook_size, embed_dim]`` (no grad).

    The adapter is frozen at construction:
      * ``requires_grad_(False)`` on every submodule.
      * ``self.train = lambda mode=True: self``  (override so Lightning's
        ``model.train()`` does not re-enable BN/dropout training inside
        the frozen tokenizer; matches official `adapter.py:54`).

    The stub encoder/decoder used here have random weights — they exist so
    the rest of the model has the correct latent shape for unit tests and
    dry-runs. For real training, instantiate with
    ``VQTokenizerAdapter.from_pretrained_compvis(vqgan_path)`` (requires
    `taming-transformers`).
    """

    def __init__(self, cfg: VQTokenizerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or VQTokenizerConfig()
        self.encoder = _StubVQGANEncoder(self.cfg)
        self.decoder = _StubVQGANDecoder(self.cfg)
        # Codebook as a buffer (not a Parameter) → never trained, never
        # touched by EMA. Matches official `quantize.VectorQuantizer.embedding`
        # except the latter is a Parameter; we use a buffer to make
        # `requires_grad_(False)` watertight even if someone forgets the
        # freeze step.
        embed = torch.randn(self.cfg.codebook_size, self.cfg.embedding_dim) * 0.02
        self.register_buffer("codebook", embed)
        self._freeze()

    # ------------------------------------------------------------------
    # freezing
    # ------------------------------------------------------------------

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        # Override `.train()` so consumers that flip the parent model into
        # train mode do not flip BN/dropout inside the tokenizer.
        self.train = lambda mode=True: self  # type: ignore[method-assign]

    # ------------------------------------------------------------------
    # codebook API
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self.cfg.embedding_dim

    @property
    def codebook_size(self) -> int:
        return self.cfg.codebook_size

    @property
    def n_tokens(self) -> int:
        return self.cfg.n_tokens

    @property
    def token_grid_size(self) -> int:
        return self.cfg.token_grid_size

    def get_codebook(self) -> torch.Tensor:
        return self.codebook.detach()

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------

    def _quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, d, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, d)
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1, keepdim=False).unsqueeze(0)
        )
        indices_flat = dist.argmin(dim=1)
        indices = indices_flat.view(b, h, w)
        quant_flat = F.embedding(indices_flat, self.codebook)
        quant = quant_flat.view(b, h, w, d).permute(0, 3, 1, 2)
        return quant, indices

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """image [B, C, H, W] → indices [B, n_tokens]."""
        z = self.encoder(image)
        _, indices = self._quantize(z)
        return indices.flatten(1)

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """indices [B, ...] → embeddings [B, ..., embed_dim] (no grad path
        beyond the embedding buffer, which is not a Parameter)."""
        return F.embedding(indices, self.codebook)

    def lookup_quant(self, indices: torch.Tensor) -> torch.Tensor:
        """indices [B, n_tokens] → quant [B, embed_dim, H, W]."""
        b = indices.shape[0]
        grid = self.cfg.token_grid_size
        embeds = self.lookup(indices.view(b, grid, grid))
        return embeds.permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        z = self.lookup_quant(indices)
        return self.decoder(z).clamp(-1.0, 1.0)

    @torch.no_grad()
    def encode_to_grid(self, image: torch.Tensor):
        """Compatibility shim for old API: (quant, indices, vq_loss=0)."""
        z = self.encoder(image)
        quant, indices = self._quantize(z)
        zero = torch.zeros((), device=image.device)
        return quant, indices, zero

    # ------------------------------------------------------------------
    # optional pretrained loader (heavy dep: taming-transformers)
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_compvis(cls, vqgan_path: str) -> VQTokenizerAdapter:
        """Load the real CompVis vq-f8-n256 checkpoint via taming-transformers.

        Expects `vqgan_path` to be a directory containing
        ``configs/config.yaml`` and ``checkpoints/model.ckpt`` — same layout
        official `VQAdapter` uses.
        """
        try:
            from omegaconf import OmegaConf
            from taming.models.vqgan import VQModel
        except ImportError as e:  # pragma: no cover — heavy optional dep
            raise ImportError(
                "Loading the real pretrained VQGAN requires "
                "`taming-transformers` and `omegaconf`. Install with: "
                "`uv add taming-transformers omegaconf`."
            ) from e
        from pathlib import Path

        path = Path(vqgan_path)
        config = OmegaConf.load(path / "configs/config.yaml").model.params
        sd = torch.load(path / "checkpoints/model.ckpt", map_location="cpu")["state_dict"]
        model = VQModel(**config)
        model.load_state_dict(sd, strict=False)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        # Wrap the real VQModel in our adapter shell.
        adapter = cls(
            VQTokenizerConfig(
                image_size=int(config.get("ddconfig", {}).get("resolution", 128)),
                in_channels=int(config.get("ddconfig", {}).get("in_channels", 3)),
                embedding_dim=int(config.get("embed_dim", 4)),
                codebook_size=int(config.get("n_embed", 256)),
                downsample_factor=8,
            )
        )
        adapter.encoder = model  # type: ignore[assignment]
        adapter.decoder = model  # type: ignore[assignment]
        adapter._pretrained = model  # type: ignore[attr-defined]
        adapter.codebook = model.quantize.embedding.weight.detach().clone()
        adapter._freeze()
        return adapter


# Backwards-compatible aliases. The old VQTokenizer / VQEncoder / VQDecoder
# names are kept so external scripts that import them keep working, but they
# now point at the adapter shell.
VQTokenizer = VQTokenizerAdapter
VQEncoder = _StubVQGANEncoder
VQDecoder = _StubVQGANDecoder


# ======================================================================
# Top-level config
# ======================================================================


@dataclass
class IFFontConfig:
    """All structural hyperparameters for the IF-Font Phase-2 model.

    Defaults match official `train.yaml` + `base.yaml`:
      * 10 decoder blocks, 8 heads, d_model 384
      * **1 self-attn + 1 cross-attn per block** (NOT 2+1)
      * dropout 0.1, no Linear bias
      * IDS max_len 35, vocab built from BabelStone + ids_iffont (radical)
      * n_refs train=4 / val=3 (official train.yaml:40 has `num_refs: 3`,
        then `+1` is added in `datasets_h5.py:185` to make it even)
      * image_size 128, in_channels 3 (RGB)
      * AR block_size = n_tokens + ids_max_len - 1 = 256 + 35 - 1 = 290
    """

    image_size: int = 128
    in_channels: int = 3
    vq: VQTokenizerConfig = field(default_factory=VQTokenizerConfig)

    # IDS encoder (now: just two embedding tables, no Transformer encoder).
    ids_vocab_size: int = 1024
    ids_max_len: int = 35

    # AR Transformer decoder
    d_model: int = 384
    n_heads: int = 8
    n_blocks: int = 10
    ffn_mult: int = 4
    dropout: float = 0.1
    bias: bool = False

    # Reference handling
    n_refs: int = 3

    @property
    def n_target_tokens(self) -> int:
        return self.vq.n_tokens

    @property
    def target_vocab_size(self) -> int:
        return self.vq.codebook_size

    @property
    def ar_block_size(self) -> int:
        """Block size for nanoGPT-style prefix AR. 256 + 35 - 1 = 290."""
        return self.n_target_tokens + self.ids_max_len - 1


# ======================================================================
# Attention building blocks (per-head QK-LN, causal/cross variants)
# ======================================================================


class _CausalSelfAttention(nn.Module):
    """nanoGPT-style causal self-attn with QK-LayerNorm.

    Official `nanogpt.CausalSelfAttention`:
      * fused QKV projection
      * per-head LN on Q and K
      * Flash SDPA (PyTorch ≥ 2.0)
    """

    def __init__(self, cfg: IFFontConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_head = cfg.n_heads
        self.n_embd = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.ln_q = nn.LayerNorm(self.head_dim)
        self.ln_k = nn.LayerNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self.ln_q(q), self.ln_k(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class _CrossAttention(nn.Module):
    """Cross-attention with separate Q-from-x and KV-from-style projections.

    Official `nanogpt.CrossAttention`. Per-head QK-LN. No causal mask.
    """

    def __init__(self, cfg: IFFontConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_head = cfg.n_heads
        self.n_embd = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        self.c_attn = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.s_attn = nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.ln_q = nn.LayerNorm(self.head_dim)
        self.ln_k = nn.LayerNorm(self.head_dim)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        b, tx, c = x.shape
        ts = style.shape[1]
        q = self.c_attn(x)
        k, v = self.s_attn(style).chunk(2, dim=-1)
        q = q.view(b, tx, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, ts, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, ts, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self.ln_q(q), self.ln_k(k)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(b, tx, c)
        return self.resid_dropout(self.c_proj(y))


class _MLP(nn.Module):
    """Standard transformer FFN (4x), GELU, with dropout."""

    def __init__(self, cfg: IFFontConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, cfg.ffn_mult * cfg.d_model, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(cfg.ffn_mult * cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class _DecoderBlock(nn.Module):
    """**1 self-attn + 1 cross-attn + 1 FFN per block** (official Block2).

    Pre-LN. The previous Phase-1 design used 2 self-attn — that was
    paper-note-wrong (see github_diff.md A1).
    """

    def __init__(self, cfg: IFFontConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = _CausalSelfAttention(cfg)
        self.ln_3 = nn.LayerNorm(cfg.d_model)
        self.attn_cross = _CrossAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = _MLP(cfg)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.attn_cross(self.ln_3(x), style)
        x = x + self.mlp(self.ln_2(x))
        return x


# ======================================================================
# IDS encoder — bare embedding tables (no Transformer encoder)
# ======================================================================


class IDSEmbedding(nn.Module):
    """Two `nn.Embedding` tables over the IDS vocabulary.

    `embedding` feeds the AR decoder as a prefix (prepended to target VQ
    embeddings).
    `embedding2` feeds the 3SA cross-attention inside the StyleEncoder.

    This is the exact design of official `encoder.IDSEncoder` once the
    commented-out Transformer encoder lines are taken at face value.
    """

    def __init__(self, vocab_size: int, n_embd: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.embedding2 = nn.Embedding(vocab_size, n_embd)

    def forward(self, ids_token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ids_token_ids: [B, L] long → (embed [B, L, n_embd], embed2 [B, L, n_embd])."""
        return self.embedding(ids_token_ids), self.embedding2(ids_token_ids)


# ======================================================================
# StyleEncoder + 3SA + MoCo wrapper
# ======================================================================


class _ConvBlock(nn.Module):
    """Small pre-activation conv block; instance-norm; reflect-pad.

    Approximates the official `modules.blocks.ConvBlock` used inside
    `QuantExtEncoder._init_enc` (`encoder.py:513-525`).
    """

    def __init__(self, c_in: int, c_out: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(c_in, affine=False)
        self.act = nn.ReLU()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(c_in, c_out, 3)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(x))
        x = self.dropout(x)
        return self.conv(self.pad(x))


class _ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = _ConvBlock(c_in, c_out, dropout=dropout)
        self.conv2 = _ConvBlock(c_out, c_out, dropout=dropout)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + self.skip(x)


class StyleEncoder(nn.Module):
    """Per-ref CNN stem + coverage-weighted global pool + 3SA cross-attn.

    Inputs (official `StyleEncoder.forward`):
      * indices: [B, N, n_tokens] long — VQ indices for N refs.
      * ids:     [B, L, c_out]    — IDS embedding (ids_embed2).
      * sim:     [B, N]           — coverage similarity per (target, ref).

    Outputs:
      * x_sss: [B, L + n_tokens, c_out] — concat([x_l, x_g]).
      * cl:    [B, c_out] — contrastive head feature (training-only).
    """

    def __init__(
        self,
        adapter: VQTokenizerAdapter,
        c_out: int,
        l_ids: int,
        *,
        n_head: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.c_out = c_out
        self.n_head = n_head
        c_in = adapter.embedding_dim  # 4

        C = 32
        self.stem = nn.Sequential(
            _ConvBlock(c_in, C, dropout=dropout),
            _ConvBlock(C * 1, C * 2, dropout=dropout),
            _ConvBlock(C * 2, C * 4, dropout=dropout),
            _ResBlock(C * 4, C * 4, dropout=dropout),
            _ResBlock(C * 4, C * 4, dropout=dropout),
            _ResBlock(C * 4, C * 8, dropout=dropout),
            _ResBlock(C * 8, c_out, dropout=dropout),
        )

        # 3SA cross-attn: IDS queries → ref-feature K/V.
        self.q_linear = nn.Linear(c_out, c_out, bias=False)
        self.kv_linear = nn.Linear(c_out, 2 * c_out, bias=False)
        self.c_proj = nn.Linear(c_out, c_out, bias=False)
        self.layer_norm = nn.LayerNorm(c_out, eps=1e-6)
        self.wpe = nn.Embedding(l_ids, c_out)
        self.ln_q = nn.LayerNorm(c_out // n_head)
        self.ln_k = nn.LayerNorm(c_out // n_head)

        # Contrastive head (replaced by MoCoWrapper for real training).
        n_tokens = adapter.n_tokens
        self.cl_head = nn.Sequential(
            nn.Linear(c_out, 1),
            nn.Flatten(-2, -1),
            nn.LayerNorm(n_tokens),
            nn.SiLU(True),
            nn.Dropout(dropout),
        )
        self.cl_fc = nn.Linear(n_tokens, c_out)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _structure_style_aggregation(
        self, ids: torch.Tensor, feat: torch.Tensor
    ) -> torch.Tensor:
        """ids: [B, L, c]; feat: [B, N*n_tokens, c] → [B, L, c]."""
        seq_len = ids.shape[1]
        pos = self.wpe(torch.arange(seq_len, device=ids.device))  # [L, c]
        ids = ids + pos.unsqueeze(0)
        b = ids.shape[0]
        head_dim = self.c_out // self.n_head

        q = self.q_linear(ids).view(b, seq_len, self.n_head, head_dim).transpose(1, 2)
        kv = self.kv_linear(feat)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(b, -1, self.n_head, head_dim).transpose(1, 2)
        v = v.view(b, -1, self.n_head, head_dim).transpose(1, 2)
        q, k = self.ln_q(q), self.ln_k(k)
        y = F.scaled_dot_product_attention(q, k, v)
        return y.transpose(1, 2).contiguous().view(b, seq_len, self.c_out)

    def forward(
        self, indices: torch.Tensor, ids: torch.Tensor, sim: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        indices: [B, N, n_tokens] long.
        ids:     [B, L, c_out].
        sim:     [B, N] — coverage similarity.
        """
        b, n_ref, n_tokens = indices.shape

        # Look up codebook quant per ref → [B*N, embed_dim, h, w].
        flat_idx = indices.view(-1, n_tokens)
        x = self.adapter.lookup_quant(flat_idx)  # [B*N, embed_dim, h, w]
        x = self.stem(x)  # [B*N, c_out, h, w]
        h_w = x.shape[-1]
        x = x.view(b, n_ref, self.c_out, h_w * h_w).permute(0, 1, 3, 2)
        # x: [B, N, n_tokens, c_out]

        # Coverage-weighted global pool.
        sim = sim * 3
        weights = sim.softmax(dim=1)  # [B, N]
        x_g = torch.einsum("bnfc,bn->bfc", x, weights)  # [B, n_tokens, c_out]

        # 3SA cross-attention.
        x_flat = x.reshape(b, n_ref * n_tokens, self.c_out)
        x_l = self._structure_style_aggregation(ids, x_flat)  # [B, L, c_out]
        x_l = self.c_proj(x_l)

        x_sss = self.layer_norm(torch.cat([x_l, x_g], dim=1))  # [B, L+n_tokens, c_out]

        if not self.training:
            return x_sss, None
        cl = self.cl_fc(self.cl_head(x_g))  # [B, c_out]
        return x_sss, cl


class MoCoWrapper(nn.Module):
    """Two StyleEncoders + momentum update + projector/predictor MLPs.

    Returns (x_sss, cl) where cl is `[B, 2, dim]` stacking (predicted query,
    momentum key) — fed into `losses.sup_cl`.
    """

    def __init__(
        self,
        adapter: VQTokenizerAdapter,
        c_out: int,
        l_ids: int,
        *,
        momentum: float = 0.995,
        mlp_dim: int = 1024,
        cl_dim: int = 256,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.momentum = momentum
        self.enc = StyleEncoder(adapter, c_out, l_ids, dropout=0.1)
        self.enc_m = StyleEncoder(adapter, c_out, l_ids, dropout=0.1)
        self._build_projector_and_predictor(c_out, mlp_dim, cl_dim)
        self._enc_sync()
        for p in self.enc_m.parameters():
            p.requires_grad = False

    def _build_mlp(
        self, num_layers: int, input_dim: int, mlp_dim: int, output_dim: int, *, last_bn: bool = True
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        for i in range(num_layers):
            d1 = input_dim if i == 0 else mlp_dim
            d2 = output_dim if i == num_layers - 1 else mlp_dim
            layers.append(nn.Linear(d1, d2, bias=False))
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(d2))
                layers.append(nn.ReLU(inplace=True))
            elif last_bn:
                layers.append(nn.BatchNorm1d(d2, affine=False))
        return nn.Sequential(*layers)

    def _build_projector_and_predictor(self, c_out: int, mlp_dim: int, dim: int) -> None:
        hidden_dim = self.enc.cl_fc.weight.shape[1]
        del self.enc.cl_fc, self.enc_m.cl_fc
        self.enc.cl_fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.enc_m.cl_fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, last_bn=False)

    @torch.no_grad()
    def _enc_sync(self) -> None:
        for pq, pk in zip(self.enc.parameters(), self.enc_m.parameters(), strict=False):
            pk.data.copy_(pq.data)

    @torch.no_grad()
    def momentum_update(self, ratio: float) -> None:
        """Cosine-scheduled momentum (matches official `MoCoWrapper.momentum_update`)."""
        m = 1.0 - 0.5 * (1.0 + math.cos(math.pi * ratio)) * (1.0 - self.momentum)
        for pq, pk in zip(self.enc.parameters(), self.enc_m.parameters(), strict=False):
            pk.data.mul_(m).add_(pq.detach().data * (1.0 - m))

    def forward(
        self, indices: torch.Tensor, ids: torch.Tensor, sim: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        sss, cl = self.enc(indices, ids, sim)
        if not self.training or cl is None:
            return sss, None
        cl_q = self.predictor(cl)
        with torch.no_grad():
            _, cl_m = self.enc_m(indices, ids, sim)
        if cl_m is None:
            cl_m = cl_q.detach()
        cl_stack = torch.stack([cl_q, cl_m.detach()], dim=1)  # [B, 2, dim]
        return sss, cl_stack


# ======================================================================
# AR Transformer decoder (nanoGPT with prefix-prepended IDS)
# ======================================================================


class TransformerARDecoder(nn.Module):
    """10-block AR decoder over target VQ token sequence with prefix-prepended IDS.

    forward(idx, style, ids_embed):
      * idx:      [B, T]    target VQ indices (training: shifted target, but
                            the shift happens in `IFFont.forward`).
      * style:    [B, Ls, d_model] (cross-attn K/V; from StyleEncoder/MoCo).
      * ids_embed:[B, Li, d_model] (prefix; from IDSEmbedding.embedding).
    Returns: logits [B, Li + T, target_vocab_size].
    """

    def __init__(self, cfg: IFFontConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.target_vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.ar_block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([_DecoderBlock(cfg) for _ in range(cfg.n_blocks)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.target_vocab_size, bias=False)
        # Weight tying (official `nanogpt.py:197`).
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # scaled init on residual c_proj weights, per nanoGPT.
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_blocks))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, style: torch.Tensor, ids_embed: torch.Tensor
    ) -> torch.Tensor:
        b, t = idx.shape
        li = ids_embed.shape[1]
        total = t + li
        assert total <= self.cfg.ar_block_size, (
            f"sequence ({total}) longer than ar_block_size ({self.cfg.ar_block_size})"
        )
        pos = torch.arange(total, dtype=torch.long, device=idx.device)
        tok = self.wte(idx)
        tok = torch.cat([ids_embed, tok], dim=1)
        pos_e = self.wpe(pos).unsqueeze(0)
        x = tok + pos_e
        # Apply dropout only to the target-token positions, not the IDS prefix
        # (official `nanogpt.GPT.forward`: drops `x[:, t:, :]` where `t` is
        # ids_embed length — note name collision with our `t`; same idea).
        x[:, li:, :] = self.drop(x[:, li:, :])
        for block in self.h:
            x = block(x, style)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        style: torch.Tensor,
        ids_embed: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        sample: bool = True,
    ) -> torch.Tensor:
        """Autoregressive sampling (matches `GPT.generate`)."""
        for _ in range(max_new_tokens):
            block_size = self.cfg.ar_block_size
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits = self(idx_cond, style, ids_embed)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(
                    logits < v[:, [-1]], torch.full_like(logits, -float("inf")), logits
                )
            probs = F.softmax(logits, dim=-1)
            if sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = probs.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ======================================================================
# Top-level model
# ======================================================================


class IFFont(nn.Module):
    """End-to-end IF-Font module (Phase 2): frozen VQGAN + IDS embeddings +
    StyleEncoder/MoCo + AR decoder.
    """

    def __init__(
        self,
        cfg: IFFontConfig,
        *,
        vq_adapter: VQTokenizerAdapter | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.vq = vq_adapter if vq_adapter is not None else VQTokenizerAdapter(cfg.vq)
        # Re-pin our config's vq to whatever the adapter actually carries.
        cfg.vq = self.vq.cfg
        self.ids_encoder = IDSEmbedding(cfg.ids_vocab_size, cfg.d_model)
        self.moco_wrapper = MoCoWrapper(
            self.vq,
            c_out=cfg.d_model,
            l_ids=cfg.ids_max_len,
        )
        self.decoder = TransformerARDecoder(cfg)

    # ------------------------------------------------------------------
    # Coverage similarity (target-vs-ref structural overlap)
    # ------------------------------------------------------------------

    @staticmethod
    def _coverage_one(target: tuple[str, ...], source: tuple[str, ...], IDC: set[str]) -> float:
        """Longest IDC-anchored common run length normalised by target len.

        Mirrors official `IDSEncoder.coverage` (`encoder.py:307-357`).
        """
        ti_max = len(target)
        if ti_max == 0:
            return 0.0
        match_cnt = 0
        ti = 0
        while ti < ti_max:
            tc = target[ti]
            if tc not in IDC:
                ti += 1
                continue
            si, si_max = 0, len(source)
            advanced = False
            while si < si_max:
                if source[si] != tc:
                    si += 1
                    continue
                ti2, si2 = ti, si
                while ti2 < ti_max and si2 < si_max and target[ti2] == source[si2]:
                    ti2 += 1
                    si2 += 1
                if ti2 == ti_max or target[ti2] in IDC:
                    match_cnt += ti2 - ti
                    ti = ti2
                    advanced = True
                    break
                si += 1
            if not advanced:
                ti += 1
        return match_cnt / ti_max

    @classmethod
    def compute_coverage(
        cls,
        target_ids: list[tuple[str, ...]],
        source_ids: list[list[tuple[str, ...]]],
        idc_chars: tuple[str, ...],
    ) -> torch.Tensor:
        """Compute coverage similarity for a batch.

        target_ids: list of length B, each a tuple of IDS tokens for the target.
        source_ids: list of length B, each a list of N tuples (ref IDS).
        idc_chars: the 12 IDC symbols.
        """
        idc_set = set(idc_chars)
        out = []
        for t, refs in zip(target_ids, source_ids, strict=False):
            row = [cls._coverage_one(t, s, idc_set) for s in refs]
            out.append(row)
        return torch.tensor(out, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode_target(self, image: torch.Tensor) -> torch.Tensor:
        """image [B, C, H, W] → indices [B, n_tokens]."""
        return self.vq.encode(image)

    def encode_refs(self, ref_images: torch.Tensor) -> torch.Tensor:
        """ref_images [B, N, C, H, W] → indices [B, N, n_tokens]."""
        b, n, c, h, w = ref_images.shape
        idx = self.vq.encode(ref_images.reshape(b * n, c, h, w))  # [B*N, n_tokens]
        return idx.view(b, n, -1)

    def forward(
        self,
        target_image: torch.Tensor,
        *,
        ids_token_ids: torch.Tensor,
        ref_images: torch.Tensor,
        coverage_sim: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute everything needed for sq (CE) + sup_cl losses.

        Args:
            target_image: [B, C, H, W] target glyph in [-1, 1].
            ids_token_ids: [B, L] long — IDS tokens (PAD-right).
            ref_images: [B, N, C, H, W] reference glyphs.
            coverage_sim: [B, N] — IDS-coverage similarity score per ref.

        Returns dict:
          * logits: [B, n_tokens, target_vocab_size] — sliced to image-token
            positions only (the official `net2net_model.py:99` slice).
          * target_ids: [B, n_tokens] — VQ indices of the target glyph.
          * cl: [B, 2, dim] or None — MoCo contrastive features (training-only).
        """
        target_ids = self.encode_target(target_image)  # [B, n_tokens]
        ref_indices = self.encode_refs(ref_images)  # [B, N, n_tokens]

        ids_embed, ids_embed2 = self.ids_encoder(ids_token_ids)  # both [B, L, d]
        x_sss, cl = self.moco_wrapper(ref_indices, ids_embed2, coverage_sim)

        # AR input is the target token sequence with prefix-prepended IDS;
        # decoder forward then slices the IDS portion out of logits, leaving
        # only image-token logits (official `net2net_model.py:98-99`):
        #
        #   logits = self.netTransformer(x[:, :-1], x_sss, embeddings=ids_embed)
        #   logits = logits[:, ids_embed.shape[1] - 1:]
        #
        # The -1 indexing on x and the +-1 slice align positions: at training
        # time we feed x[:-1] and the decoder predicts shifted-by-one tokens.
        logits = self.decoder(target_ids[:, :-1], x_sss, ids_embed)  # [B, L+T-1, K]
        ids_len = ids_embed.shape[1]
        logits = logits[:, ids_len - 1:]  # [B, T, K]

        return {
            "logits": logits,  # [B, n_tokens, target_vocab_size]
            "target_ids": target_ids,
            "cl": cl,
            "x_sss": x_sss,
            "ids_embed": ids_embed,
        }

    @torch.no_grad()
    def sample(
        self,
        *,
        ids_token_ids: torch.Tensor,
        ref_images: torch.Tensor,
        coverage_sim: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = 100,
        sample: bool = True,
    ) -> torch.Tensor:
        """Autoregressive sample → image [B, C, H, W]."""
        ref_indices = self.encode_refs(ref_images)
        ids_embed, ids_embed2 = self.ids_encoder(ids_token_ids)
        was_training = self.moco_wrapper.training
        self.moco_wrapper.eval()
        x_sss, _ = self.moco_wrapper(ref_indices, ids_embed2, coverage_sim)
        if was_training:
            self.moco_wrapper.train()

        b = ids_token_ids.shape[0]
        device = ids_token_ids.device
        idx0 = torch.empty(b, 0, dtype=torch.long, device=device)
        seq = self.decoder.generate(
            idx0,
            x_sss,
            ids_embed,
            max_new_tokens=self.cfg.n_target_tokens,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
        )
        seq = seq.clamp(0, self.cfg.target_vocab_size - 1)
        return self.vq.decode_indices(seq)


def build_if_font(cfg: IFFontConfig, *, vq_adapter: VQTokenizerAdapter | None = None) -> IFFont:
    return IFFont(cfg, vq_adapter=vq_adapter)


# Stable public re-exports (back-compat aliases for old VectorQuantizer etc.
# are intentionally dropped; the previous in-training EMA codebook does not
# exist any more).
class VectorQuantizer(nn.Module):  # pragma: no cover — deprecated stub kept for import compat
    """DEPRECATED. The Phase-2 IF-Font uses a frozen pretrained tokenizer;
    in-training VQ updates no longer exist. This stub keeps old imports
    from breaking but raises if anyone actually constructs it."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "VectorQuantizer is deprecated in Phase 2 — IF-Font uses a frozen "
            "pretrained CompVis vq-f8-n256 adapter (see VQTokenizerAdapter). "
            "Update your code to load via `VQTokenizerAdapter.from_pretrained_compvis` "
            "or accept the random-weight stub for tests."
        )


# Silence unused-import warnings for the deterministic-init helper.
_ = random


__all__ = [
    "IDSEmbedding",
    "IFFont",
    "IFFontConfig",
    "MoCoWrapper",
    "StyleEncoder",
    "TransformerARDecoder",
    "VQDecoder",
    "VQEncoder",
    "VQTokenizer",
    "VQTokenizerAdapter",
    "VQTokenizerConfig",
    "VectorQuantizer",
    "build_if_font",
]
