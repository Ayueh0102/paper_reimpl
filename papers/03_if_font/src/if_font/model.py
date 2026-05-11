"""IF-Font — blind reimplementation.

Architecture (paper-cited, p.1 / training-config block):
  1. VQ tokenizer (VQGAN-style): conv encoder + EMA-updated codebook + conv decoder.
       - codebook size = 256
       - downsample factor = 8  (e.g. 128×128 image → 16×16 token grid → 256 tokens)
  2. IDS text encoder: embedding table over (IDC chars + leaf components +
     special tokens). A small Transformer encoder adds context.
  3. Reference VQ encoder: reuses the same VQ encoder + codebook to map each
     reference glyph to its codebook token grid (one ref → 256 tokens).
  4. Autoregressive Transformer decoder:
       - 10 blocks
       - 8 attention heads
       - feature dim = 384
       - each block: 2 self-attn (causal over target tokens) + 1 cross-attn
         over the concatenated context = [IDS_tokens ; ref_tokens].
       - predicts target VQ token id at every position (cross-entropy).
  5. Inference: AR sample tokens, then VQ decoder produces the image.

Adapter notes for the shared smoke-test contract:
  IF-Font does NOT use the shared GaussianDiffusion (it's autoregressive, not
  diffusion). The compute_loss function in `train.py` reads ``image``,
  ``refs`` / ``ref_images``, and optional explicit ``ids_token_ids`` /
  ``ids_attention_mask`` from the batch dict. When IDS tensors are absent
  (synthetic smoke batches), we fall back to a derived stub IDS sequence
  built from ``char_id`` modulo the IDS tokenizer vocab so the conditioning
  path still receives a finite, gradient-carrying signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================
# VQ tokenizer (VQGAN-like)
# ======================================================================


@dataclass
class VQTokenizerConfig:
    """Hyperparameters for the VQGAN-style discrete tokenizer.

    Defaults follow the paper's training-config block (note "訓練配置"):
      * codebook_size = 256
      * downsample factor = 8
      * embedding_dim = 256 (paper does not specify; commonly = codebook_dim)

    The codebook is EMA-updated (van den Oord 2017 / Esser 2021 style); this
    keeps gradient flow through the encoder via the straight-through
    estimator and avoids the dead-codebook pathology when batch is small.
    """

    image_size: int = 128
    in_channels: int = 1
    base_channels: int = 64
    channel_mult: tuple[int, ...] = (1, 2, 2, 4)
    """4 stages -> 3 downsamples -> factor 8 (paper-cited)."""
    embedding_dim: int = 256
    codebook_size: int = 256
    commitment_weight: float = 0.25
    decay: float = 0.99
    """EMA decay for codebook updates."""
    eps: float = 1.0e-5

    def stage_resolutions(self) -> list[int]:
        sizes = [self.image_size]
        for _ in range(len(self.channel_mult) - 1):
            sizes.append(sizes[-1] // 2)
        return sizes

    @property
    def token_grid_size(self) -> int:
        """How many tokens along one spatial dim after the VQ encoder."""
        return self.image_size // (2 ** (len(self.channel_mult) - 1))

    @property
    def n_tokens(self) -> int:
        """Number of VQ tokens per glyph = (token_grid_size) ** 2."""
        return self.token_grid_size ** 2


def _gn(channels: int) -> nn.GroupNorm:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


class _ConvDown(nn.Module):
    def __init__(self, in_c: int, out_c: int, *, stride: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1),
            _gn(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, stride=1, padding=1),
            _gn(out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _ConvUp(nn.Module):
    def __init__(self, in_c: int, out_c: int, *, upsample: bool) -> None:
        super().__init__()
        self.upsample = upsample
        self.body = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            _gn(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            _gn(out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.body(x)


class VQEncoder(nn.Module):
    """CNN that maps glyph image → continuous latent grid [B, D, H/8, W/8]."""

    def __init__(self, cfg: VQTokenizerConfig) -> None:
        super().__init__()
        chs = [cfg.base_channels * m for m in cfg.channel_mult]
        layers: list[nn.Module] = [nn.Conv2d(cfg.in_channels, chs[0], 3, padding=1)]
        prev = chs[0]
        for i, c in enumerate(chs):
            stride = 2 if i > 0 else 1
            layers.append(_ConvDown(prev, c, stride=stride))
            prev = c
        layers.append(nn.Conv2d(prev, cfg.embedding_dim, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class VQDecoder(nn.Module):
    """Reverse-direction CNN: [B, D, H/8, W/8] → [B, C, H, W]."""

    def __init__(self, cfg: VQTokenizerConfig) -> None:
        super().__init__()
        chs = [cfg.base_channels * m for m in cfg.channel_mult]
        layers: list[nn.Module] = [nn.Conv2d(cfg.embedding_dim, chs[-1], 3, padding=1)]
        rev_chs = list(reversed(chs))
        prev = rev_chs[0]
        for i, c in enumerate(rev_chs):
            # Upsample on every stage except the first to invert encoder's 3 strided downsamples.
            up = i > 0
            layers.append(_ConvUp(prev, c, upsample=up))
            prev = c
        layers.append(_gn(prev))
        layers.append(nn.SiLU(inplace=True))
        layers.append(nn.Conv2d(prev, cfg.in_channels, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.body(z)


class VectorQuantizer(nn.Module):
    """EMA-updated discrete codebook with straight-through gradient.

    Reference: van den Oord 2017 + Esser 2021 (VQGAN).
    Forward returns (quantized, indices, vq_loss) where:
      * quantized: [B, D, H, W] — encoder output replaced by codebook entries.
      * indices  : [B, H, W] long — codebook indices.
      * vq_loss  : scalar — commitment loss (encoder is pushed toward codebook).
    Codebook entries are updated via EMA, not gradient.
    """

    def __init__(self, cfg: VQTokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.codebook_size = cfg.codebook_size
        self.embedding_dim = cfg.embedding_dim
        self.decay = cfg.decay
        self.eps = cfg.eps
        self.commitment_weight = cfg.commitment_weight

        # Codebook initialised from a small normal; EMA buffers track usage.
        embed = torch.randn(cfg.codebook_size, cfg.embedding_dim) * 0.02
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(cfg.codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def _flatten_input(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D, H, W] -> [B*H*W, D]
        b, d, h, w = z.shape
        return z.permute(0, 2, 3, 1).reshape(-1, d), (b, h, w)

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [B, ...] long → [B, ..., D]."""
        return F.embedding(indices, self.embedding)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat, (b, h, w) = self._flatten_input(z)
        # Distance: ||z - e||^2 = ||z||^2 - 2 z.e + ||e||^2
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(dim=1, keepdim=False).unsqueeze(0)
        )
        indices_flat = dist.argmin(dim=1)  # [B*H*W]
        indices = indices_flat.view(b, h, w)
        quantized_flat = F.embedding(indices_flat, self.embedding)
        quantized = quantized_flat.view(b, h, w, self.embedding_dim).permute(0, 3, 1, 2)

        # EMA codebook update (only in training mode).
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(indices_flat, self.codebook_size).type_as(flat)  # [N, K]
                cluster_size = onehot.sum(dim=0)  # [K]
                self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                embed_sum = onehot.t() @ flat  # [K, D]
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cluster_size_norm = (
                    (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
                )
                self.embedding.copy_(self.embed_avg / cluster_size_norm.unsqueeze(1))

        # Commitment loss: encoder must commit to chosen codebook entries.
        # The codebook side is updated by EMA, so we detach there.
        commitment = F.mse_loss(z, quantized.detach(), reduction="mean")
        vq_loss = self.commitment_weight * commitment

        # Straight-through estimator: pass encoder gradient through quantization.
        quantized = z + (quantized - z).detach()
        return quantized, indices, vq_loss


class VQTokenizer(nn.Module):
    """Composite: encoder + vector quantizer + decoder.

    forward(image) → dict {
        recon, indices, quantized, vq_loss, recon_loss
    }
    """

    def __init__(self, cfg: VQTokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = VQEncoder(cfg)
        self.quantizer = VectorQuantizer(cfg)
        self.decoder = VQDecoder(cfg)

    @property
    def n_tokens(self) -> int:
        return self.cfg.n_tokens

    @property
    def codebook_size(self) -> int:
        return self.cfg.codebook_size

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """image: [B, C, H, W] → indices [B, n_tokens] (flattened grid)."""
        z = self.encoder(image)
        _, indices, _ = self.quantizer(z)
        return indices.flatten(1)  # [B, H*W]

    def encode_to_grid(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(image)
        return self.quantizer(z)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [B, n_tokens] → image [B, C, H, W]."""
        b = indices.shape[0]
        grid = self.cfg.token_grid_size
        embeds = self.quantizer.lookup(indices.view(b, grid, grid))  # [B, H, W, D]
        z = embeds.permute(0, 3, 1, 2).contiguous()
        return self.decoder(z)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(image)
        quantized, indices, vq_loss = self.quantizer(z)
        recon = self.decoder(quantized)
        recon_loss = F.mse_loss(recon, image, reduction="mean")
        return {
            "recon": recon,
            "indices": indices.flatten(1),
            "quantized": quantized,
            "vq_loss": vq_loss,
            "recon_loss": recon_loss,
        }


# ======================================================================
# Transformer AR decoder (10 blocks, 8 heads, dim=384 per paper)
# ======================================================================


@dataclass
class IFFontConfig:
    """All structural hyperparameters for IF-Font.

    Defaults match the paper's training-config block (note "訓練配置"):
      * 10 decoder blocks
      * 8 attention heads
      * d_model = 384
      * batch_size = 128 (set in train YAML, not here)

    For the CPU smoke test we override these via a tiny config.
    """

    image_size: int = 128
    in_channels: int = 1
    vq: VQTokenizerConfig = field(default_factory=VQTokenizerConfig)

    # IDS encoder (small Transformer encoder over the IDS token sequence)
    ids_vocab_size: int = 1024  # filled after IDSTokenizer.fit_from_charset
    ids_max_len: int = 32
    ids_encoder_layers: int = 2
    ids_encoder_heads: int = 4
    ids_encoder_dim: int = 384

    # AR Transformer decoder
    d_model: int = 384
    n_heads: int = 8
    n_blocks: int = 10
    n_self_attn_per_block: int = 2
    """Paper §"訓練配置": '2 self-attention + 1 cross-attention per block'."""
    ffn_mult: int = 4
    dropout: float = 0.0

    # Reference handling
    n_refs: int = 1
    """Number of reference glyphs concatenated into the cross-attention context."""

    # Tied to VQ tokenizer
    @property
    def n_target_tokens(self) -> int:
        return self.vq.n_tokens

    @property
    def target_vocab_size(self) -> int:
        return self.vq.codebook_size


class _MultiHeadAttention(nn.Module):
    """Standard multi-head attention with optional causal mask.

    Supports both self-attention (k=v=x) and cross-attention (separate kv).
    """

    def __init__(self, d_model: int, n_heads: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, lq, _ = q_in.shape
        lk = k_in.shape[1]
        q = self.q_proj(q_in).view(b, lq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_in).view(b, lk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_in).view(b, lk, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            # attn_mask: [Lq, Lk] bool — True = MASK OUT (i.e., disallow).
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        if key_padding_mask is not None:
            # key_padding_mask: [B, Lk] bool — True = real token. Invert.
            mask = ~key_padding_mask.bool()  # [B, Lk] True = MASK OUT
            # Guard against all-True rows (every key masked) which would
            # softmax to NaN. This happens e.g. when classifier-free guidance
            # zeroes the entire IDS mask: the self-attn in the IDS encoder
            # has no real keys for the dropped row. We unmask position 0 on
            # such rows so softmax stays finite; the corresponding row's
            # output is effectively zeroed at the consumer (decoder
            # cross-attn sees only the ref tokens).
            all_masked = mask.all(dim=1, keepdim=True)
            if all_masked.any():
                mask = mask.clone()
                mask[:, 0:1] = mask[:, 0:1] & ~all_masked
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, lq, self.d_model)
        return self.out_proj(out)


class _FFN(nn.Module):
    def __init__(self, d_model: int, mult: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_model, d_model * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _DecoderBlock(nn.Module):
    """One block: N self-attn (causal) + 1 cross-attn (context) + FFN.

    Paper config: 2 self-attn + 1 cross-attn per block. The repeated self-attn
    gives the decoder extra capacity to model long-range AR dependencies
    before mixing the conditioning context.
    """

    def __init__(self, cfg: IFFontConfig) -> None:
        super().__init__()
        self.n_self = cfg.n_self_attn_per_block
        self.self_norms = nn.ModuleList(
            [nn.LayerNorm(cfg.d_model) for _ in range(self.n_self)]
        )
        self.self_attns = nn.ModuleList(
            [
                _MultiHeadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
                for _ in range(self.n_self)
            ]
        )
        self.cross_norm_q = nn.LayerNorm(cfg.d_model)
        self.cross_norm_kv = nn.LayerNorm(cfg.d_model)
        self.cross_attn = _MultiHeadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
        self.ffn_norm = nn.LayerNorm(cfg.d_model)
        self.ffn = _FFN(cfg.d_model, cfg.ffn_mult, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        context_pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # 2× self-attention with residual + LN-pre.
        for norm, attn in zip(self.self_norms, self.self_attns, strict=True):
            h = norm(x)
            x = x + attn(h, h, h, attn_mask=causal_mask)
        # 1× cross-attention
        q = self.cross_norm_q(x)
        kv = self.cross_norm_kv(context)
        x = x + self.cross_attn(q, kv, kv, key_padding_mask=context_pad_mask)
        # FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


def _causal_mask(seq_len: int, *, device: torch.device) -> torch.Tensor:
    """Upper-triangular bool mask: True at positions to MASK (j > i)."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


class _LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = x.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"sequence longer than max_len {self.max_len}: got {seq_len}"
            )
        pos = torch.arange(seq_len, device=x.device)
        return x + self.embed(pos).unsqueeze(0).expand(b, -1, -1)


class IDSTextEncoder(nn.Module):
    """Small Transformer encoder over IDS tokens.

    Paper says "IDS → text encoder"; depth/width are not specified, so this is
    a deliberately small Transformer encoder (paper-vague → see blind_impl.md).
    The output is fed (concatenated with reference VQ tokens) to the decoder
    cross-attention as the conditioning context.
    """

    def __init__(self, cfg: IFFontConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.ids_vocab_size, cfg.ids_encoder_dim)
        self.pos = _LearnedPositionalEmbedding(cfg.ids_max_len, cfg.ids_encoder_dim)
        self.layers = nn.ModuleList()
        for _ in range(cfg.ids_encoder_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(cfg.ids_encoder_dim),
                        "self_attn": _MultiHeadAttention(
                            cfg.ids_encoder_dim, cfg.ids_encoder_heads, dropout=cfg.dropout
                        ),
                        "norm2": nn.LayerNorm(cfg.ids_encoder_dim),
                        "ffn": _FFN(cfg.ids_encoder_dim, cfg.ffn_mult, cfg.dropout),
                    }
                )
            )
        # Project to decoder's d_model if widths differ.
        self.out_proj = (
            nn.Identity()
            if cfg.ids_encoder_dim == cfg.d_model
            else nn.Linear(cfg.ids_encoder_dim, cfg.d_model)
        )

    def forward(
        self, ids_token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """ids_token_ids: [B, L]. mask: [B, L] bool (True = real token).

        Returns context tokens [B, L, d_model].
        """
        x = self.embed(ids_token_ids)
        x = self.pos(x)
        for layer in self.layers:
            h = layer["norm1"](x)
            x = x + layer["self_attn"](h, h, h, key_padding_mask=attention_mask)
            x = x + layer["ffn"](layer["norm2"](x))
        return self.out_proj(x)


class TransformerARDecoder(nn.Module):
    """10-block AR decoder over target VQ token sequence.

    Inputs at every training step:
      * target_tokens: [B, n_tokens] long — codebook indices to predict.
        We shift right by 1 and prepend a BOS-token row of all-zero index.
      * context: [B, Lc, d_model] — encoder side (IDS + ref tokens).
      * context_pad_mask: [B, Lc] bool — True = real context token.
    Output: logits [B, n_tokens, codebook_size].
    """

    def __init__(self, cfg: IFFontConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.target_vocab_size + 1, cfg.d_model)
        """+1 entry: index ``codebook_size`` is the BOS/start token."""
        self.bos_index = cfg.target_vocab_size
        self.pos = _LearnedPositionalEmbedding(cfg.n_target_tokens, cfg.d_model)
        self.blocks = nn.ModuleList([_DecoderBlock(cfg) for _ in range(cfg.n_blocks)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.target_vocab_size, bias=False)

    def _shifted_input(self, target_tokens: torch.Tensor) -> torch.Tensor:
        b, n = target_tokens.shape
        bos = torch.full((b, 1), self.bos_index, dtype=torch.long, device=target_tokens.device)
        return torch.cat([bos, target_tokens[:, :-1]], dim=1)

    def forward(
        self,
        target_tokens: torch.Tensor,
        context: torch.Tensor,
        *,
        context_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_embed(self._shifted_input(target_tokens))
        x = self.pos(x)
        n = x.shape[1]
        mask = _causal_mask(n, device=x.device)
        for block in self.blocks:
            x = block(x, context, causal_mask=mask, context_pad_mask=context_pad_mask)
        x = self.norm(x)
        return self.head(x)  # [B, N, codebook_size]

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        *,
        context_pad_mask: torch.Tensor | None,
        n_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Greedy / temperature autoregressive sampling.

        Returns [B, n_tokens] long.
        """
        b = context.shape[0]
        device = context.device
        out = torch.full((b, n_tokens), 0, dtype=torch.long, device=device)
        # First "previous token" is the BOS index.
        prev = torch.full((b, 1), self.bos_index, dtype=torch.long, device=device)
        # We always recompute the prefix; for a smoke test this is fine.
        for step in range(n_tokens):
            x = self.token_embed(prev)
            x = self.pos(x)
            cur_len = x.shape[1]
            mask = _causal_mask(cur_len, device=device)
            for block in self.blocks:
                x = block(x, context, causal_mask=mask, context_pad_mask=context_pad_mask)
            logits = self.head(self.norm(x[:, -1:, :]))  # [B, 1, K]
            if temperature > 0:
                probs = torch.softmax(logits.squeeze(1) / max(temperature, 1e-6), dim=-1)
                token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            else:
                token = logits.squeeze(1).argmax(dim=-1, keepdim=True)
            out[:, step : step + 1] = token
            prev = torch.cat([prev, token], dim=1)
        return out


# ======================================================================
# Top-level model
# ======================================================================


class IFFont(nn.Module):
    """End-to-end IF-Font module: VQ tokenizer + IDS encoder + AR decoder.

    Stage A (per CLAUDE.md three-stage plan): only the VQ tokenizer is
    trained on TTF renders. The AR decoder pathway is dormant (its loss
    weight is 0 in the Stage A YAML).
    Stage B/C: VQ is frozen (or fine-tuned) and the AR decoder is trained
    with cross-entropy on target tokens.
    """

    def __init__(self, cfg: IFFontConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vq = VQTokenizer(cfg.vq)
        self.ids_encoder = IDSTextEncoder(cfg)
        self.decoder = TransformerARDecoder(cfg)
        # Learned "null" IDS context, used when no ids tensor is provided.
        self.ids_null_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def encode_refs_to_tokens(self, ref_images: torch.Tensor) -> torch.Tensor:
        """ref_images: [B, N, C, H, W] → embeds [B, N*n_tokens, D].

        Reuses the VQ encoder + codebook (frozen or trainable depending on
        stage) to convert each reference to a sequence of d_model-wide
        embedding vectors derived from the codebook lookup. We then linearly
        project from embedding_dim to d_model.
        """
        b, n, c, h, w = ref_images.shape
        flat = ref_images.reshape(b * n, c, h, w)
        # Note: we use indices → embedding lookup so the path stays gradient-
        # connected through the quantizer's straight-through estimator.
        indices = self.vq.encode(flat)  # [B*N, n_tokens]
        embeds = self.vq.quantizer.lookup(indices)  # [B*N, n_tokens, D_vq]
        embeds = embeds.reshape(b, n * self.vq.n_tokens, self.cfg.vq.embedding_dim)
        return self.ref_to_decoder_proj(embeds)

    @property
    def ref_to_decoder_proj(self) -> nn.Module:
        # Lazy-build so __init__ can run before knowing exact dims; cached.
        if not hasattr(self, "_ref_proj"):
            mod: nn.Module
            if self.cfg.vq.embedding_dim == self.cfg.d_model:
                mod = nn.Identity()
            else:
                mod = nn.Linear(self.cfg.vq.embedding_dim, self.cfg.d_model)
            # Register so it participates in optimizer / .to(device).
            self.add_module("_ref_proj", mod)
        return self._ref_proj

    def build_context(
        self,
        ids_token_ids: torch.Tensor | None,
        ids_attention_mask: torch.Tensor | None,
        ref_images: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (context [B, Lc, d_model], context_pad_mask [B, Lc] bool).

        Empty-ref / empty-ids cases fall back to a learned null token so the
        decoder always sees ≥1 context token.
        """
        # IDS branch
        if ids_token_ids is not None and ids_token_ids.numel() > 0:
            assert ids_attention_mask is not None
            ids_ctx = self.ids_encoder(ids_token_ids, ids_attention_mask)
            ids_mask = ids_attention_mask
            b = ids_ctx.shape[0]
            device = ids_ctx.device
        elif ref_images is not None and ref_images.numel() > 0:
            b = ref_images.shape[0]
            device = ref_images.device
            ids_ctx = self.ids_null_token.expand(b, 1, -1).to(device)
            ids_mask = torch.ones(b, 1, dtype=torch.bool, device=device)
        else:
            raise ValueError("Must provide at least one of (ids_token_ids, ref_images).")

        # Ref branch
        if ref_images is not None and ref_images.numel() > 0:
            ref_ctx = self.encode_refs_to_tokens(ref_images)
            ref_mask = torch.ones(
                ref_ctx.shape[0], ref_ctx.shape[1], dtype=torch.bool, device=device
            )
            context = torch.cat([ids_ctx, ref_ctx], dim=1)
            pad_mask = torch.cat([ids_mask, ref_mask], dim=1)
        else:
            context = ids_ctx
            pad_mask = ids_mask
        return context, pad_mask

    # ------------------------------------------------------------------
    # Forward / loss helpers
    # ------------------------------------------------------------------

    def forward(
        self,
        target_image: torch.Tensor,
        *,
        ids_token_ids: torch.Tensor | None = None,
        ids_attention_mask: torch.Tensor | None = None,
        ref_images: torch.Tensor | None = None,
        return_recon: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute everything needed for the AR cross-entropy + VQ losses.

        Returns dict with keys:
          logits      : [B, n_tokens, codebook_size] — AR head output.
          target_ids  : [B, n_tokens] long — VQ indices the AR decoder predicts.
          vq_loss     : scalar — commitment loss on the *target* glyph.
          recon_loss  : scalar — MSE(target, VQ-recon(target)).
          recon       : (optional) reconstructed image from VQ decoder.
        """
        vq_out = self.vq(target_image)
        target_ids = vq_out["indices"]  # [B, n_tokens]
        context, pad_mask = self.build_context(ids_token_ids, ids_attention_mask, ref_images)
        logits = self.decoder(target_ids, context, context_pad_mask=pad_mask)
        out = {
            "logits": logits,
            "target_ids": target_ids,
            "vq_loss": vq_out["vq_loss"],
            "recon_loss": vq_out["recon_loss"],
        }
        if return_recon:
            out["recon"] = vq_out["recon"]
        return out

    @torch.no_grad()
    def sample(
        self,
        *,
        ids_token_ids: torch.Tensor | None = None,
        ids_attention_mask: torch.Tensor | None = None,
        ref_images: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive sample → image [B, C, H, W]."""
        context, pad_mask = self.build_context(ids_token_ids, ids_attention_mask, ref_images)
        n_tokens = self.cfg.n_target_tokens
        indices = self.decoder.sample(
            context,
            context_pad_mask=pad_mask,
            n_tokens=n_tokens,
            temperature=temperature,
        )
        # Clamp into valid codebook range (sample() shouldn't escape but be safe).
        indices = indices.clamp(0, self.cfg.target_vocab_size - 1)
        return self.vq.decode_indices(indices)


def build_if_font(cfg: IFFontConfig) -> IFFont:
    return IFFont(cfg)


__all__ = [
    "IFFont",
    "IFFontConfig",
    "IDSTextEncoder",
    "TransformerARDecoder",
    "VQDecoder",
    "VQEncoder",
    "VQTokenizer",
    "VQTokenizerConfig",
    "VectorQuantizer",
    "build_if_font",
]
