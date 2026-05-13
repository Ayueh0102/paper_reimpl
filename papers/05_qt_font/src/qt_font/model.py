"""QT-Font model — Phase 2 redesign.

The Phase 1 blind impl was a fixed-depth saturated quadtree D3PM over 8-bin
pixel intensities. The official `lsflyt-pku/QT-Font` repo is a 2-D port of
DualOctreeGNN over a contour+skeleton point cloud with K=3 axis labels, T=1000
cosine-schedule D3PM uniform, multi-depth split CE supervision, and graph
U-Nets over content/style octrees as conditioning. This module ships the
paper-aligned version of all of those, written in pure PyTorch on top of the
minimal sparse octree in :mod:`qt_font.octree`.

Conceptual stack
----------------
1. **Octree**: glyph image → ``{0=bg, 1=contour, 2=skeleton}`` label map → sparse
   octree (dense down to ``full_depth``, adaptive past it). See
   :func:`qt_font.octree.build_octree_from_image`.

2. **D3PM Uniform** with K=3 classes, T=1000, cosine (Glide) β schedule.
   Forward marginals are computed exactly in O(K²) per step.

3. **Graph U-Net** with edge-direction-aware GraphConv (5 edge types =
   N/E/S/W + self) operating on each depth's sparse node list. Time and
   conditioning are injected per resblock per depth, mirroring
   ``third_party/05_qt_font/models/graph_diffusion.py:148-149``.

4. **Per-depth split heads** (2-way at inner depths, 3-way at the leaf depth)
   that supervise the multi-depth CE loss in ``losses.compute_multi_depth_ce``.

Top-level entry points
----------------------
* :class:`QTFontConfig`  — dataclass of hyper-params (see Phase 2 train YAML).
* :class:`QTFontModel`   — the full model. Operates on octrees (NOT pixel tensors).
* :class:`D3PMUniform`   — discrete diffusion utility (forward q_sample + Q caches).
* :func:`build_qt_font`  — factory.

Backwards compatibility shim
----------------------------
The Phase 1 ``model.forward(x_t, t, content=..., char_id=..., ...)`` pixel
adapter is preserved for the shared smoke harness; internally it converts the
pixel tensor into an octree, runs the new model, and renders the leaf labels
back to a pixel image. The native training loss uses the octree path
directly — pixels are only a convenience boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .octree import (
    OctreeBatch,
    build_octree_from_image,
    render_label_image,
)

# --------------------------------------------------------------------------- #
# D3PM Uniform with cosine schedule (paper schedule_deltas S1, S2, S3).        #
# --------------------------------------------------------------------------- #


class D3PMUniform(nn.Module):
    """Discrete diffusion with uniform transition matrix and cosine β schedule.

    Mirrors ``third_party/05_qt_font/datasets/chinesefont_asymmetric.py:55-81``
    and ``third_party/05_qt_font/main.py:166-184`` — both compute exactly the
    same ``Q``, ``Q_T`` and cumulative ``Q_`` tensors. We keep the official
    parameterisation::

        Q[t, i, i] = 1 - β_t · (K-1) / K
        Q[t, i, j != i] = β_t / K

    which keeps every row sum exactly 1 for any K (Phase 1's
    ``(1-β_t)·I + (β_t/K)·11ᵀ`` parameterisation drifts row sums for K > 2).

    Buffers
    -------
    betas         : (T,) cosine-schedule β.
    Q             : (T, K, K) one-step transition.
    Q_cum         : (T, K, K) cumulative transition Q̄_t = Π_{s≤t} Q_s.
    Q_T           : (T, K, K) Q transposed (handy for q_posterior_logits).
    """

    def __init__(
        self,
        *,
        n_states: int = 3,
        timesteps: int = 1000,
        schedule: str = "cos",
        beta_start: float = 0.02,
        beta_end: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_states = int(n_states)
        self.timesteps = int(timesteps)
        self.schedule = schedule

        if schedule == "cos":
            # Glide cosine schedule:
            #   ᾱ_t = cos²((t/T + 0.008) / 1.008 · π/2)
            #   β_t = min(1 - ᾱ_t / ᾱ_{t-1}, 0.999)
            steps = np.arange(self.timesteps + 1, dtype=np.float64) / self.timesteps
            alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
            betas_np = np.minimum(1.0 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
            betas = torch.tensor(betas_np, dtype=torch.float32)
        elif schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        else:  # pragma: no cover
            raise ValueError(f"unknown schedule={schedule!r}")

        K = self.n_states
        Q = torch.ones(self.timesteps, K, K, dtype=torch.float32)
        for i in range(K):
            for j in range(K):
                if i == j:
                    Q[:, i, j] = 1.0 - betas * (K - 1) / K
                else:
                    Q[:, i, j] = betas / K

        Q_T = Q.permute(0, 2, 1).contiguous()
        Q_cum = torch.ones(self.timesteps, K, K, dtype=torch.float32)
        Q_cum[0] = Q[0]
        for t in range(1, self.timesteps):
            Q_cum[t] = Q_cum[t - 1] @ Q[t]

        # All buffers — they ride along on `.to(device)` but are deterministic
        # functions of the schedule so we keep them non-persistent.
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("Q", Q, persistent=False)
        self.register_buffer("Q_T", Q_T, persistent=False)
        self.register_buffer("Q_cum", Q_cum, persistent=False)

    def q_probs(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``q(x_t | x_0)`` categorical probabilities. ``x0`` (N,), ``t`` (N,)."""
        # Q_cum[t] is (K, K); we want the x0-th row → (N, K).
        return self.Q_cum[t, x0]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample ``x_t ~ q(x_t | x_0)`` via Gumbel-max on log q_probs.

        Mirrors ``third_party/05_qt_font/datasets/chinesefont_asymmetric.py:102-112``.
        Gumbel-max is exact and avoids the ``torch.multinomial`` reshape dance.
        """
        probs = self.q_probs(x0, t)  # (N, K)
        logits = torch.log(probs + 1e-8)
        gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-8, 1.0)))
        return torch.argmax(logits + gumbel, dim=-1)

    def sample_random_step(self, n: int, device: torch.device | str) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (n,), device=device, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Building blocks: time embedding, edge-typed GraphConv, GraphResBlock.        #
# --------------------------------------------------------------------------- #


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding + 2-layer MLP with Swish activation.

    Mirrors ``third_party/05_qt_font/models/graph_diffusion.py:26-44``.
    """

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        if n_channels < 8 or n_channels % 8 != 0:
            raise ValueError(f"n_channels={n_channels} must be >= 8 and divisible by 8")
        self.n_channels = n_channels
        self.lin1 = nn.Linear(n_channels // 4, n_channels)
        self.lin2 = nn.Linear(n_channels, n_channels)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / max(1, half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # Swish: x * sigmoid(x).
        emb = self.lin1(emb)
        emb = emb * torch.sigmoid(emb)
        return self.lin2(emb)


class EdgeTypedGraphConv(nn.Module):
    """Edge-direction-aware graph convolution.

    Equivalent to the per-edge-type weight stacking in
    ``third_party/05_qt_font/models/modules_bn.py:66-110``: each of the 4
    directional edge types (N/E/S/W) has its own weight matrix that is applied
    to the source node's feature before aggregation; the 5th "self" type is
    realised as a residual ``W_self · x`` on the destination.

    Forward
    -------
    x          : (N, C_in)
    edge_index : (2, E) long
    edge_type  : (E,)   long in [0, 3]
    Returns    : (N, C_out)
    """

    def __init__(self, in_dim: int, out_dim: int, n_edge_types: int = 4) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_edge_types = n_edge_types
        # One weight per directional edge type + one for the self loop.
        self.edge_weights = nn.Parameter(torch.empty(n_edge_types, in_dim, out_dim))
        self.self_weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.kaiming_uniform_(self.edge_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.self_weight, a=math.sqrt(5))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        out = x @ self.self_weight  # (N, out_dim) — self loop
        if edge_index.numel() > 0:
            # Sparse impl: process each edge_type in turn so we never
            # materialise the per-edge weight tensor (E, in_dim, out_dim).
            # Each per-type matmul is (E_t, in) @ (in, out) → (E_t, out)
            # — peak intermediate ~ E_t * out, vs the dense E * in * out.
            # For depth=7 (16k+ edges) this drops peak memory by ~in_dim×.
            for et in range(self.n_edge_types):
                mask = edge_type == et
                if not bool(mask.any()):
                    continue
                sub_src = edge_index[0][mask]
                sub_dst = edge_index[1][mask]
                x_sub = x[sub_src]                       # (E_t, in_dim)
                msg = x_sub @ self.edge_weights[et]      # (E_t, out_dim)
                out.index_add_(0, sub_dst, msg)
        return out + self.bias


class GraphResBlock(nn.Module):
    """Pre-activation graph residual block with timestep + cond conditioning.

    Mirrors ``GraphResBlocks(.., resblock_type='basic')`` in
    ``third_party/05_qt_font/models/modules_bn.py``: BN → SiLU → GraphConv →
    AddTimeCond → BN → SiLU → GraphConv → +skip.
    """

    def __init__(self, dim: int, cond_dim: int, n_edge_types: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(8, dim), num_channels=dim)
        self.conv1 = EdgeTypedGraphConv(dim, dim, n_edge_types=n_edge_types)
        self.cond_proj = nn.Linear(cond_dim, dim)
        self.norm2 = nn.GroupNorm(num_groups=min(8, dim), num_channels=dim)
        self.conv2 = EdgeTypedGraphConv(dim, dim, n_edge_types=n_edge_types)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        cond_per_node: torch.Tensor,
    ) -> torch.Tensor:
        # GroupNorm expects (N, C) reshaped to (1, C, N) — group_norm operates
        # per-channel, so we transpose, apply, transpose back.
        def _gn(norm: nn.GroupNorm, h: torch.Tensor) -> torch.Tensor:
            return norm(h.t().unsqueeze(0)).squeeze(0).t()

        h = _gn(self.norm1, x)
        h = F.silu(h)
        h = self.conv1(h, edge_index, edge_type)
        h = h + self.cond_proj(F.silu(cond_per_node))
        h = _gn(self.norm2, h)
        h = F.silu(h)
        h = self.conv2(h, edge_index, edge_type)
        return x + h


# --------------------------------------------------------------------------- #
# Conditioning: graph encoders for content + style octrees.                    #
# --------------------------------------------------------------------------- #


class OctreeEncoder(nn.Module):
    """Graph U-Net encoder for a content / style reference octree.

    Mirrors the "style_conv + style_encoder + style_downsample" stack in
    ``third_party/05_qt_font/models/graph_diffusion.py:69-80``. The encoder
    produces a feature per node at the bottleneck depth (``depth_stop``).
    """

    def __init__(
        self,
        *,
        in_feat_dim: int,
        channels_per_depth: dict[int, int],
        full_depth: int,
        depth: int,
        depth_stop: int,
        n_edge_types: int = 4,
        cond_dim: int = 64,
    ) -> None:
        super().__init__()
        self.full_depth = full_depth
        self.depth = depth
        self.depth_stop = depth_stop
        # Project input features into channel width at the leaf depth.
        self.input_conv = EdgeTypedGraphConv(
            in_feat_dim, channels_per_depth[depth], n_edge_types=n_edge_types
        )
        # One resblock per depth from depth → depth_stop (inclusive).
        self.blocks = nn.ModuleDict()
        self.downs = nn.ModuleDict()
        for d in range(depth, depth_stop - 1, -1):
            self.blocks[str(d)] = GraphResBlock(channels_per_depth[d], cond_dim, n_edge_types)
            if d > depth_stop:
                # Downsample: aggregate from depth d into depth d-1 via parent index.
                self.downs[str(d)] = nn.Linear(channels_per_depth[d], channels_per_depth[d - 1])

    def forward(
        self, octree: OctreeBatch, cond_dummy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Run the encoder.

        Parameters
        ----------
        octree : OctreeBatch
        cond_dummy : (B, cond_dim)
            Time/conditioning vector; the encoders don't really use it (they
            run before time is meaningful) but the GraphResBlock signature
            wants something — we feed zeros. Phase 4 cleanup: split the block.
        """
        feats: dict[int, torch.Tensor] = {}
        # Build per-node input feature: (row/side - 0.5, col/side - 0.5, 1) — a
        # normalised position + an occupancy flag. This is the moral equivalent
        # of the 4-channel ``Points(features=cat(xy, one_hot_axis))`` input the
        # official repo passes to its style_conv (cf. ``main.py:336``).
        leaf = octree.levels[octree.depth]
        side = 1 << octree.depth
        pos = leaf.xy.float() / max(1, side - 1) - 0.5  # (N, 2)
        occ = torch.ones((leaf.xy.shape[0], 1), device=leaf.xy.device, dtype=pos.dtype)
        input_feat = torch.cat([pos, occ], dim=-1)
        # Pad to whatever in_feat_dim was registered for.
        if input_feat.shape[-1] < self.input_conv.in_dim:
            pad = torch.zeros(
                (input_feat.shape[0], self.input_conv.in_dim - input_feat.shape[-1]),
                device=input_feat.device,
                dtype=input_feat.dtype,
            )
            input_feat = torch.cat([input_feat, pad], dim=-1)

        h = self.input_conv(input_feat, leaf.edge_index, leaf.edge_type)
        for d in range(self.depth, self.depth_stop - 1, -1):
            lvl = octree.levels[d]
            cond_node = cond_dummy[lvl.batch_id]
            h = self.blocks[str(d)](h, lvl.edge_index, lvl.edge_type, cond_node)
            feats[d] = h
            if d > self.depth_stop:
                # Pool to parent: scatter-mean h into the (d-1) node list using
                # the lvl.parent index.
                parent = lvl.parent
                parent_lvl = octree.levels[d - 1]
                pooled = torch.zeros(
                    (parent_lvl.xy.shape[0], h.shape[-1]),
                    device=h.device,
                    dtype=h.dtype,
                )
                counts = torch.zeros(
                    (parent_lvl.xy.shape[0],), device=h.device, dtype=h.dtype
                )
                pooled.index_add_(0, parent, h)
                counts.index_add_(0, parent, torch.ones_like(parent, dtype=h.dtype))
                pooled = pooled / counts.clamp_min(1.0).unsqueeze(-1)
                h = self.downs[str(d)](pooled)
        return feats


# --------------------------------------------------------------------------- #
# Top-level config + model.                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class ConditioningBundle:
    """Container for conditioning paths.

    The paper-aligned model takes octrees for content + style; the pixel
    adapter (smoke / shared infra) provides image tensors that we lift into
    octrees on the fly.

    Backwards-compat fields (``char_id``, ``writer_id``, ``script_id``, etc.)
    are ACCEPTED so existing call sites do not break, but they are no longer
    consumed by the model — see paper conditioning_deltas C1.
    """

    content: torch.Tensor | None = None  # (B, 1, H, W) image
    refs: torch.Tensor | None = None  # (B, R, 1, H, W) image stack
    # Legacy fields — ignored.
    char_id: torch.Tensor | None = None
    script_id: torch.Tensor | None = None
    writer_id: torch.Tensor | None = None
    style_family_id: torch.Tensor | None = None
    unit_id: torch.Tensor | None = None
    ref_images: torch.Tensor | None = None  # alias of `refs` from the smoke harness
    ref_valid: torch.Tensor | None = None

    def resolved_refs(self) -> torch.Tensor | None:
        return self.refs if self.refs is not None else self.ref_images


@dataclass
class QTFontConfig:
    """Configuration for :class:`QTFontModel` — Phase 2 paper-aligned defaults."""

    # Geometry.
    image_size: int = 128
    full_depth: int = 4  # depths [0, full_depth] are dense
    depth: int = 7  # leaf depth (128 px → 7; 256 px → 8)
    depth_stop: int = 4  # bottleneck depth in the U-Net
    n_states: int = 3  # {bg, contour, skeleton}

    # Channels per depth. Default mirrors the 128/256 px config in
    # ``third_party/05_qt_font/models/graph_diffusion.py:421``.
    channels_per_depth: tuple[int, ...] = (3, 512, 512, 256, 512, 256, 128, 64, 64, 64)
    cond_dim: int = 256  # time + content/style vector width

    # Diffusion.
    timesteps: int = 1000
    schedule: str = "cos"

    # Conditioning.
    use_style: bool = True
    use_content: bool = True

    # Backwards-compat fields (paper drops these — kept for shared harness).
    in_channels: int = 1
    content_channels: int = 1
    ref_channels: int = 1
    char_vocab_size: int = 0
    writer_vocab_size: int = 0
    script_vocab_size: int = 0
    hidden_dim: int = 128  # unused; superseded by channels_per_depth
    n_layers: int = 2
    time_embed_dim: int = 256  # alias of cond_dim for legacy YAML keys

    # Misc.
    extras: dict = field(default_factory=dict)


class QTFontModel(nn.Module):
    """Paper-aligned QT-Font diffusion model.

    Forward pass
    ------------
    1. Inner-octree feature lift from ``leaf.xy`` → graph U-Net encoder.
    2. Bottleneck mid-blocks.
    3. Graph U-Net decoder with per-depth predict heads emitting:
       - 2-way "split / no-split" logits at every inner depth
       - 3-way "{bg, contour, skeleton}" logits at the leaf depth
    4. Loss is computed externally over ``logits_per_depth`` against the GT
       octree's ``split`` + ``leaf_label`` fields. (See :mod:`qt_font.losses`.)
    """

    def __init__(self, cfg: QTFontConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._validate()
        self._build_modules()

    # ------------------------------------------------------------------ #

    def _validate(self) -> None:
        c = self.cfg
        if c.full_depth >= c.depth:
            raise ValueError(f"full_depth ({c.full_depth}) must be < depth ({c.depth})")
        if c.depth_stop < c.full_depth or c.depth_stop > c.depth:
            raise ValueError(
                f"depth_stop ({c.depth_stop}) must be in [full_depth, depth]="
                f"[{c.full_depth}, {c.depth}]"
            )
        if len(c.channels_per_depth) <= c.depth:
            raise ValueError(
                f"channels_per_depth has {len(c.channels_per_depth)} entries but depth={c.depth}"
            )
        if c.n_states != 3:
            raise ValueError(
                f"n_states must be 3 for the paper-aligned axis labels (got {c.n_states})"
            )

    def _build_modules(self) -> None:
        c = self.cfg
        chans = {d: c.channels_per_depth[d] for d in range(c.depth + 1)}

        # Time embedding emits (B, cond_dim).
        self.time_embed = TimeEmbedding(c.cond_dim)

        # Style + content encoders share the OctreeEncoder shape but have
        # independent weights — matches the official ``style_*`` / ``content_*``
        # split (graph_diffusion.py:69-80).
        encoder_in_feat = 3  # (row, col, occ)
        if c.use_style:
            self.style_enc = OctreeEncoder(
                in_feat_dim=encoder_in_feat,
                channels_per_depth=chans,
                full_depth=c.full_depth,
                depth=c.depth,
                depth_stop=c.depth_stop,
                cond_dim=c.cond_dim,
            )
            self.style_fc = nn.Linear(2 * chans[c.depth_stop], c.cond_dim // 2)
        if c.use_content:
            self.content_enc = OctreeEncoder(
                in_feat_dim=encoder_in_feat,
                channels_per_depth=chans,
                full_depth=c.full_depth,
                depth=c.depth,
                depth_stop=c.depth_stop,
                cond_dim=c.cond_dim,
            )
            self.content_fc = nn.Linear(2 * chans[c.depth_stop], c.cond_dim // 2)
        if not (c.use_style or c.use_content):
            # Fully unconditional — register an empty learnable cond so the
            # rest of the pipeline doesn't have to special-case None.
            self.uncond_cond = nn.Parameter(torch.zeros(c.cond_dim))

        # The input octree (noisy) is lifted with an in_feat_dim of 3 same as
        # the cond encoders: (row, col, occ); we *also* concatenate the one-hot
        # axis label at the leaf depth so the model knows which of K labels the
        # noisy sample carries. That makes the total input feat width
        # (3 + K) = 6 at the leaf depth.
        self.input_conv = EdgeTypedGraphConv(3 + c.n_states, chans[c.depth])

        # U-Net encoder: depth → depth_stop.
        self.enc_blocks = nn.ModuleDict()
        self.enc_downs = nn.ModuleDict()
        for d in range(c.depth, c.depth_stop - 1, -1):
            self.enc_blocks[str(d)] = GraphResBlock(chans[d], c.cond_dim)
            if d > c.depth_stop:
                self.enc_downs[str(d)] = nn.Linear(chans[d], chans[d - 1])

        # Bottleneck.
        self.mid_block1 = GraphResBlock(chans[c.depth_stop], c.cond_dim)
        self.mid_block2 = GraphResBlock(chans[c.depth_stop], c.cond_dim)

        # U-Net decoder.
        self.dec_blocks = nn.ModuleDict()
        self.dec_ups = nn.ModuleDict()
        for d in range(c.depth_stop, c.depth + 1):
            self.dec_blocks[str(d)] = GraphResBlock(chans[d], c.cond_dim)
            if d > c.depth_stop:
                self.dec_ups[str(d)] = nn.Linear(chans[d - 1], chans[d])

        # Per-depth predict heads. 2 logits at inner depths, K at leaf depth.
        self.predict_heads = nn.ModuleDict()
        for d in range(c.depth_stop, c.depth + 1):
            out_dim = c.n_states if d == c.depth else 2
            self.predict_heads[str(d)] = nn.Sequential(
                nn.Linear(chans[d], 32),
                nn.SiLU(),
                nn.Linear(32, out_dim),
            )

    # ------------------------------------------------------------------ #
    # Conditioning.                                                       #
    # ------------------------------------------------------------------ #

    def _encode_cond_octree(
        self,
        octree: OctreeBatch,
        encoder: OctreeEncoder,
        fc: nn.Linear,
        cond_dummy: torch.Tensor,
    ) -> torch.Tensor:
        """Run encoder, then maxpool+avgpool across nodes per batch → cond vec."""
        feats = encoder(octree, cond_dummy)
        h = feats[encoder.depth_stop]  # (N, C)
        batch_id = octree.levels[encoder.depth_stop].batch_id
        # Per-batch max + mean pool. ``scatter_max`` is not in pure-PyTorch core,
        # so we do it via a loop over the batch — fine for B ≤ 16.
        B = octree.batch_size
        C = h.shape[-1]
        max_per_b = torch.zeros((B, C), device=h.device, dtype=h.dtype)
        mean_per_b = torch.zeros((B, C), device=h.device, dtype=h.dtype)
        for b in range(B):
            mask = batch_id == b
            if mask.any():
                hb = h[mask]
                max_per_b[b] = hb.max(dim=0).values
                mean_per_b[b] = hb.mean(dim=0)
        pooled = torch.cat([max_per_b, mean_per_b], dim=-1)
        return fc(pooled)

    def encode_conditioning(
        self,
        timesteps: torch.Tensor,
        *,
        content_octree: OctreeBatch | None = None,
        style_octrees: list[OctreeBatch] | None = None,
    ) -> torch.Tensor:
        """Fuse (timesteps, content, style) → (B, cond_dim).

        Style across multiple reference glyphs is mean-pooled in feature space,
        matching ``third_party/05_qt_font/models/graph_diffusion.py:316-322``.
        """
        c = self.cfg
        t_emb = self.time_embed(timesteps)  # (B, cond_dim)
        cond_parts: list[torch.Tensor] = []

        # Use zeros as the dummy condition for the inner encoders.
        zero_cond = torch.zeros_like(t_emb)

        if c.use_content and content_octree is not None:
            content_vec = self._encode_cond_octree(
                content_octree, self.content_enc, self.content_fc, zero_cond
            )
            cond_parts.append(content_vec)
        else:
            cond_parts.append(torch.zeros((timesteps.shape[0], c.cond_dim // 2), device=timesteps.device))

        if c.use_style and style_octrees is not None and len(style_octrees) > 0:
            style_vecs = [
                self._encode_cond_octree(soct, self.style_enc, self.style_fc, zero_cond)
                for soct in style_octrees
            ]
            style_vec = torch.stack(style_vecs, dim=0).mean(dim=0)
            cond_parts.append(style_vec)
        else:
            cond_parts.append(torch.zeros((timesteps.shape[0], c.cond_dim // 2), device=timesteps.device))

        cond = torch.cat(cond_parts, dim=-1)  # (B, cond_dim)
        return t_emb + cond

    # ------------------------------------------------------------------ #
    # Octree forward — multi-depth logits.                                #
    # ------------------------------------------------------------------ #

    def predict_logits(
        self,
        octree: OctreeBatch,
        cond: torch.Tensor,
        *,
        noisy_leaf_label: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        """Run encoder → bottleneck → decoder → per-depth predict heads.

        Parameters
        ----------
        octree : OctreeBatch
            **Topology comes from this octree** (typically the GT octree during
            training). The model never has to predict topology at training
            time — it only predicts (split / no-split) at inner depths and
            (3-class axis) at the leaf depth, mirroring the official setup
            where ``octree_out = batch['octree_gt']`` during training
            (``third_party/05_qt_font/main.py:58``).
        cond : (B, cond_dim)
            Conditioning vector; broadcast to per-node via ``batch_id``,
            mirroring ``graph_diffusion.py:148-149``.
        noisy_leaf_label : LongTensor (N_leaf,) | None
            The noisy 3-class label at the leaf depth (``q_sample(x0, t)``).
            If ``None``, the GT ``octree.levels[depth].leaf_label`` is used —
            which is fine for smoke tests but is **not** the diffusion training
            path. The real loss uses noisy labels here.

        Returns
        -------
        logits_per_depth : dict[int, Tensor]
            ``logits_per_depth[d]`` has shape ``(N_d, K)`` where K is 2 at
            inner depths (split vs no-split) and ``n_states`` (=3) at the leaf
            depth (axis label).
        """
        c = self.cfg

        # Build the leaf input feature: (row, col, occ, one_hot_label).
        leaf = octree.levels[octree.depth]
        side = 1 << octree.depth
        pos = leaf.xy.float() / max(1, side - 1) - 0.5
        occ = torch.ones((leaf.xy.shape[0], 1), device=pos.device, dtype=pos.dtype)
        label = noisy_leaf_label if noisy_leaf_label is not None else leaf.leaf_label
        if label is None:
            label = torch.zeros((leaf.xy.shape[0],), dtype=torch.long, device=leaf.xy.device)
        one_hot = F.one_hot(label, num_classes=c.n_states).float()
        input_feat = torch.cat([pos, occ, one_hot], dim=-1)

        h = self.input_conv(input_feat, leaf.edge_index, leaf.edge_type)

        # Encoder.
        enc_feats: dict[int, torch.Tensor] = {}
        for d in range(c.depth, c.depth_stop - 1, -1):
            lvl = octree.levels[d]
            cond_node = cond[lvl.batch_id]
            h = self.enc_blocks[str(d)](h, lvl.edge_index, lvl.edge_type, cond_node)
            enc_feats[d] = h
            if d > c.depth_stop:
                parent = lvl.parent
                parent_lvl = octree.levels[d - 1]
                pooled = torch.zeros(
                    (parent_lvl.xy.shape[0], h.shape[-1]),
                    device=h.device,
                    dtype=h.dtype,
                )
                counts = torch.zeros(
                    (parent_lvl.xy.shape[0],), device=h.device, dtype=h.dtype
                )
                pooled.index_add_(0, parent, h)
                counts.index_add_(0, parent, torch.ones_like(parent, dtype=h.dtype))
                pooled = pooled / counts.clamp_min(1.0).unsqueeze(-1)
                h = self.enc_downs[str(d)](pooled)

        # Bottleneck (two extra resblocks at depth_stop).
        bottleneck_lvl = octree.levels[c.depth_stop]
        cond_node = cond[bottleneck_lvl.batch_id]
        h = self.mid_block1(h, bottleneck_lvl.edge_index, bottleneck_lvl.edge_type, cond_node)
        h = self.mid_block2(h, bottleneck_lvl.edge_index, bottleneck_lvl.edge_type, cond_node)

        # Decoder.
        logits_per_depth: dict[int, torch.Tensor] = {}
        for d in range(c.depth_stop, c.depth + 1):
            lvl = octree.levels[d]
            cond_node = cond[lvl.batch_id]
            if d > c.depth_stop:
                # Upsample: each fine node gets its parent's coarse feature.
                parent = lvl.parent
                parent_feat = h[parent]
                h = self.dec_ups[str(d)](parent_feat)
                # Skip from encoder.
                if d in enc_feats and enc_feats[d].shape == h.shape:
                    h = h + enc_feats[d]
            h = self.dec_blocks[str(d)](h, lvl.edge_index, lvl.edge_type, cond_node)
            logits_per_depth[d] = self.predict_heads[str(d)](h)

        return logits_per_depth

    # ------------------------------------------------------------------ #
    # Pixel-space adapter (for shared smoke harness).                     #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        content: torch.Tensor | None = None,
        ref_images: torch.Tensor | None = None,
        refs: torch.Tensor | None = None,
        # Legacy id kwargs (accepted, ignored).
        char_id: torch.Tensor | None = None,
        writer_id: torch.Tensor | None = None,
        script_id: torch.Tensor | None = None,
        style_family_id: torch.Tensor | None = None,
        unit_id: torch.Tensor | None = None,
        ref_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pixel-in / pixel-out adapter.

        Used only by the shared smoke harness and dry-run probes. Internally:
        ``x_t`` (B,1,H,W) → extract 3-class labels → build sparse octree →
        run :meth:`predict_logits` → render leaf-argmax back to a pixel image.

        For the *native* training loss, prefer driving :meth:`predict_logits`
        directly with octrees built by the dataset.
        """
        c = self.cfg
        # Build the noisy octree from x_t. Content/style optional.
        octree = build_octree_from_image(
            x_t, full_depth=c.full_depth, depth=c.depth
        ).to(x_t.device)
        content_octree = None
        if content is not None and c.use_content:
            # If content is multi-channel, take the first channel.
            if content.dim() == 4 and content.shape[1] != 1:
                content = content[:, :1]
            content_octree = build_octree_from_image(
                content, full_depth=c.full_depth, depth=c.depth
            ).to(x_t.device)
        refs_in = refs if refs is not None else ref_images
        style_octrees: list[OctreeBatch] | None = None
        if refs_in is not None and c.use_style:
            # refs: (B, R, C, H, W). Build one octree per reference position.
            R = refs_in.shape[1]
            style_octrees = []
            for r in range(R):
                ref_r = refs_in[:, r, :1]  # take first channel
                style_octrees.append(
                    build_octree_from_image(
                        ref_r, full_depth=c.full_depth, depth=c.depth
                    ).to(x_t.device)
                )
        cond = self.encode_conditioning(
            timesteps,
            content_octree=content_octree,
            style_octrees=style_octrees,
        )
        logits_per_depth = self.predict_logits(octree, cond)
        # Render: take argmax at the leaf depth and map {0,1,2}→{-1, 0, +1}.
        leaf_logits = logits_per_depth[c.depth]
        labels = leaf_logits.argmax(dim=-1)
        leaf_lvl = octree.levels[c.depth]
        side = 1 << c.depth
        B = octree.batch_size
        img = torch.zeros((B, side, side), device=x_t.device)
        # Map labels back to a continuous gray image: bg→-1, contour→0, skeleton→+1.
        mapping = torch.tensor([-1.0, 0.0, 1.0], device=x_t.device)
        values = mapping[labels]
        img[leaf_lvl.batch_id, leaf_lvl.xy[:, 0], leaf_lvl.xy[:, 1]] = values
        img = img.unsqueeze(1)  # (B, 1, side, side)
        if c.image_size != side:
            img = F.interpolate(img, size=(c.image_size, c.image_size), mode="bilinear", align_corners=False)
        return img


def build_qt_font(cfg: QTFontConfig) -> QTFontModel:
    return QTFontModel(cfg)


# --------------------------------------------------------------------------- #
# Legacy exports kept so the package surface doesn't break.                    #
# --------------------------------------------------------------------------- #


def quantize_to_states(*_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        "quantize_to_states was Phase 1's K=8 intensity-bin path; the Phase 2 "
        "redesign uses extract_glyph_labels + build_octree_from_image."
    )


def decode_states_to_image(*_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        "decode_states_to_image was Phase 1's adapter; use render_label_image "
        "from qt_font.octree instead."
    )


def build_quadtree_states(*_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        "build_quadtree_states was the Phase 1 full-saturated tree path; the "
        "Phase 2 redesign uses build_octree_from_image (adaptive sparse)."
    )


__all__ = [
    "ConditioningBundle",
    "D3PMUniform",
    "QTFontConfig",
    "QTFontModel",
    "build_qt_font",
    "render_label_image",
]
