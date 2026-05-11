"""QT-Font model: Dual quadtree graph network + discrete diffusion.

Blind reimplementation — see ``reports/blind_impl.md`` for the decision log.

Conceptual stack
----------------
1. ``quantize_to_states`` : pixel image (B,1,H,W) in [-1,1]
   → per-leaf categorical states (B, L) over K bins, where L = 4^depth.
2. ``D3PMUniform``       : adds uniform discrete noise to the leaf states for a
   given timestep, producing ``x_t`` (one-hot or class indices).
3. ``QTFontModel``       : the dual quadtree graph U-Net. Takes noisy node states
   plus conditioning (timestep, content, char_id, writer_id, ref_images) and
   returns per-leaf state logits ∈ R^{B,L,K} as the discrete reverse process.
4. ``decode_states_to_image`` : project predicted leaf states back into a pixel
   image so that the shared sampler / smoke harness can consume it.

The standard entrypoint signature is preserved:
    QTFontModel.forward(x_t, timesteps, content=..., char_id=..., writer_id=...,
                        ref_images=..., ref_valid=..., ...) -> pixel tensor

``x_t`` is a pixel tensor; the wrapper does the quadtree state conversion
internally so that the model is drop-in for the shared smoke / sampler infra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# Quadtree fan-out: every internal node has exactly four children (NE/NW/SE/SW).
# Surfaces explicitly so the constant is grep-able and reviewable in one place.
_QUAD = 4

# --------------------------------------------------------------------------- #
# Quadtree construction                                                       #
# --------------------------------------------------------------------------- #


def _full_quadtree_offsets(depth: int) -> tuple[list[int], list[int]]:
    """Return (level_start, level_size) offsets for a full quadtree of `depth`.

    Level 0 = single root, level d = 4^d nodes. Total nodes = (4^(d+1)-1)/3.
    """
    level_size = [_QUAD**lvl for lvl in range(depth + 1)]
    level_start: list[int] = [0]
    for s in level_size[:-1]:
        level_start.append(level_start[-1] + s)
    return level_start, level_size


def _build_parent_child_index(depth: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build flat parent/child index tensors for a full quadtree.

    Returns
    -------
    parent_of : LongTensor, shape (N,)
        parent_of[i] = index of parent node (root maps to -1 sentinel).
    child_of  : LongTensor, shape (N, 4)
        child_of[i] = 4 child indices (padded with -1 for leaves).
    """
    level_start, level_size = _full_quadtree_offsets(depth)
    total = level_start[-1] + level_size[-1]
    parent_of = torch.full((total,), -1, dtype=torch.long)
    child_of = torch.full((total, _QUAD), -1, dtype=torch.long)
    for lvl in range(depth):
        s = level_start[lvl]
        n = level_size[lvl]
        s_next = level_start[lvl + 1]
        for k in range(n):
            cur = s + k
            for c in range(_QUAD):
                ch = s_next + k * _QUAD + c
                child_of[cur, c] = ch
                parent_of[ch] = cur
    return parent_of, child_of


def _build_leaf_sibling_index(depth: int, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build leaf spatial-neighbor index tensors (4-connectivity within leaf grid).

    Returns
    -------
    leaf_neighbors : LongTensor (L, 4)  — N/E/S/W neighbour leaf id (-1 on border)
    leaf_xy        : LongTensor (L, 2)  — (row, col) of each leaf in the 2^d × 2^d grid
    """
    grid = 2**depth
    leaf_neighbors = torch.full((grid * grid, _QUAD), -1, dtype=torch.long)
    leaf_xy = torch.zeros((grid * grid, 2), dtype=torch.long)
    for r in range(grid):
        for c in range(grid):
            idx = r * grid + c
            leaf_xy[idx, 0] = r
            leaf_xy[idx, 1] = c
            if r > 0:
                leaf_neighbors[idx, 0] = (r - 1) * grid + c
            if c + 1 < grid:
                leaf_neighbors[idx, 1] = r * grid + (c + 1)
            if r + 1 < grid:
                leaf_neighbors[idx, 2] = (r + 1) * grid + c
            if c > 0:
                leaf_neighbors[idx, 3] = r * grid + (c - 1)
    return leaf_neighbors, leaf_xy


def quantize_to_states(image: torch.Tensor, *, depth: int, n_states: int) -> torch.Tensor:
    """Pixel image (B,1,H,W) in [-1,1] → per-leaf class indices (B, L).

    Strategy: split image into 2^depth × 2^depth tiles; mean-pool each tile;
    map [-1,1] → {0,...,K-1} by uniform binning. This is the natural fully
    saturated quadtree of fixed depth.

    Notes
    -----
    For batched, fixed-shape tensors we use a *full* (saturated) quadtree
    rather than the paper's adaptive sparse one. The expressivity is roughly
    equivalent at this depth, and it makes the model trainable with stock
    PyTorch ops. See blind_impl.md.
    """
    if image.dim() != 4:
        raise ValueError(f"expected (B,1,H,W), got {tuple(image.shape)}")
    grid = 2**depth
    B, _C, H, W = image.shape
    if H % grid != 0 or W % grid != 0:
        raise ValueError(f"image size {H}x{W} not divisible by 2^depth={grid}")
    pooled = F.adaptive_avg_pool2d(image, (grid, grid))  # (B, C, grid, grid)
    pooled = pooled.mean(dim=1)  # (B, grid, grid) treat content as 1-channel
    # Map [-1, 1] → [0, K-1] integer
    normalized = ((pooled + 1.0) * 0.5).clamp(0.0, 1.0 - 1e-6)
    states = (normalized * n_states).long()
    return states.reshape(B, grid * grid)


def decode_states_to_image(
    state_logits: torch.Tensor,
    *,
    depth: int,
    n_states: int,
    image_size: int,
) -> torch.Tensor:
    """Per-leaf softmax over K classes → pixel image (B, 1, H, W).

    The decoded pixel = E[class index] mapped back to [-1, 1].
    Then bilinearly upsampled from (2^depth) grid to image_size.
    """
    B, L, K = state_logits.shape
    grid = 2**depth
    if L != grid * grid:
        raise ValueError(f"leaf count {L} mismatch with depth {depth} (expected {grid*grid})")
    if K != n_states:
        raise ValueError(f"n_states={n_states} mismatch with logits last dim {K}")
    probs = F.softmax(state_logits, dim=-1)
    bin_centers = torch.linspace(-1.0, 1.0, n_states + 1, device=probs.device)
    bin_centers = (bin_centers[:-1] + bin_centers[1:]) * 0.5
    expected = (probs * bin_centers).sum(dim=-1)  # (B, L)
    expected_grid = expected.reshape(B, 1, grid, grid)
    if image_size == grid:
        return expected_grid
    return F.interpolate(expected_grid, size=(image_size, image_size), mode="bilinear", align_corners=False)


def build_quadtree_states(
    image: torch.Tensor, *, depth: int, n_states: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose ``quantize_to_states`` with parent/child index tensors.

    Returns
    -------
    leaf_states : (B, L) long
    parent_of   : (N,)   long
    child_of    : (N, 4) long
    """
    leaf_states = quantize_to_states(image, depth=depth, n_states=n_states)
    parent_of, child_of = _build_parent_child_index(depth)
    return leaf_states, parent_of, child_of


# --------------------------------------------------------------------------- #
# Discrete diffusion (D3PM Uniform)                                           #
# --------------------------------------------------------------------------- #


class D3PMUniform(nn.Module):
    """Discrete diffusion with uniform transition matrix.

    Following Austin et al. 2021 (D3PM). The transition matrix at step t is::

        Q_t = (1 - β_t) · I + β_t / K · 1·1^T

    Cumulative ``Q̄_t = Π_{s<=t} Q_s`` has closed form for uniform schedules::

        Q̄_t[i, j] = ᾱ_t · 1[i==j] + (1 - ᾱ_t) / K

    where ᾱ_t = Π (1 - β_s). We only need q(x_t | x_0) which is a single
    categorical sample.

    The schedule tensors ``betas`` and ``alphas_cumprod`` are registered as
    buffers so they move with ``.to(device)`` and do not require a per-step
    cross-device copy inside :meth:`q_probs`. ``t`` may be drawn from
    ``[0, T-1]``; ``t=0`` means "one step of noise added", not "clean x_0",
    matching the sampler convention of iterating ``reversed(range(T))``.
    """

    def __init__(
        self,
        *,
        n_states: int,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.n_states = int(n_states)
        self.timesteps = int(timesteps)
        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # Register as non-persistent buffers: they move with `.to(device)` but
        # are deterministic functions of (timesteps, beta_start, beta_end) so
        # there's no need to persist them in state_dict.
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)

    def q_probs(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """q(x_t | x_0) as a categorical probability tensor.

        Parameters
        ----------
        x0 : LongTensor (B, L) — clean class indices.
        t  : LongTensor (B,)   — diffusion step per sample.

        Returns
        -------
        probs : Tensor (B, L, K)
        """
        K = self.n_states
        # ``alphas_cumprod`` lives on the same device as the module after the
        # caller does ``diffusion.to(device)``; assert rather than silently
        # paying a per-step host↔device transfer.
        if self.alphas_cumprod.device != x0.device:
            raise RuntimeError(
                f"D3PMUniform device {self.alphas_cumprod.device} != x0 device {x0.device}. "
                "Call `diffusion.to(x0.device)` once at construction."
            )
        alpha_bar = self.alphas_cumprod.gather(0, t)  # (B,)
        # alpha_bar.view(-1, 1, 1) broadcasts over the leaf dim (L) and the
        # state dim (K) of the (B, L, K) one-hot tensor below.
        alpha_bar = alpha_bar.view(-1, 1, 1)
        one_hot = F.one_hot(x0, num_classes=K).float()  # (B, L, K)
        return alpha_bar * one_hot + (1.0 - alpha_bar) / K

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0)."""
        probs = self.q_probs(x0, t)
        B, L, K = probs.shape
        flat = probs.reshape(B * L, K)
        sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
        return sampled.reshape(B, L)

    def sample_random_step(self, batch: int, device: torch.device | str) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch,), device=device, dtype=torch.long)

    def loss_x0_ce(self, logits: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Cross-entropy on the x_0 prediction (D3PM uses this auxiliary loss).

        ``logits`` shape (B, L, K); ``x0`` shape (B, L).
        """
        _B, _L, K = logits.shape
        return F.cross_entropy(logits.reshape(-1, K), x0.reshape(-1))


# --------------------------------------------------------------------------- #
# Building blocks                                                              #
# --------------------------------------------------------------------------- #


def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal time embedding (Vaswani-style)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class GraphConv(nn.Module):
    """Simple message-passing layer on a fixed adjacency.

    For each node, aggregate features from neighbours (mean) and combine with
    self features through an MLP. Adjacency is provided as an index tensor of
    shape ``(N, K_neigh)`` where -1 = no neighbour.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.neigh_proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, neigh_idx: torch.Tensor) -> torch.Tensor:
        """x: (B, N, C); neigh_idx: (N, K). Returns (B, N, out)."""
        B, N, C = x.shape
        K = neigh_idx.shape[1]
        # Build a mask & safe index (replace -1 with 0; mask later).
        valid = (neigh_idx >= 0).float().unsqueeze(0).unsqueeze(-1)  # (1, N, K, 1)
        idx = neigh_idx.clamp_min(0)  # (N, K)
        # gather: (B, N, K, C)
        gathered = x[:, idx.reshape(-1), :].reshape(B, N, K, C)
        masked = gathered * valid
        denom = valid.sum(dim=2).clamp_min(1.0)
        neigh_mean = masked.sum(dim=2) / denom
        out = self.self_proj(x) + self.neigh_proj(neigh_mean)
        return F.gelu(self.norm(out))


class ContentAwarePool(nn.Module):
    """Aggregate fine (leaf) features into a coarser graph with learned gates.

    For each parent, pool features from its 4 children weighted by a learned
    saliency softmax over the children. This is our blind interpretation of the
    paper's "content-aware pooling" — paper says the gate depends on node
    content; we let the gate be a small MLP over the child feature vector.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, leaf_x: torch.Tensor, child_of_parents: torch.Tensor) -> torch.Tensor:
        """leaf_x: (B, L, C); child_of_parents: (P, 4) leaf ids of each parent.

        Returns (B, P, C) aggregated parent features.
        """
        B, _L, C = leaf_x.shape
        P, K = child_of_parents.shape
        idx = child_of_parents.clamp_min(0).reshape(-1)
        gathered = leaf_x[:, idx, :].reshape(B, P, K, C)
        valid = (child_of_parents >= 0).float().unsqueeze(0).unsqueeze(-1)
        scores = self.score(gathered).squeeze(-1)  # (B, P, K)
        scores = scores.masked_fill(valid.squeeze(-1) == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, P, K, 1)
        return (gathered * attn).sum(dim=2)


class StyleEncoder(nn.Module):
    """Tiny CNN style encoder for a stack of reference glyphs.

    Input: (B, R, C, H, W) — R reference images.
    Output: (B, D) pooled style vector.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embed_dim = embed_dim

    def forward(self, refs: torch.Tensor, ref_valid: torch.Tensor | None = None) -> torch.Tensor:
        if refs.dim() == 4:
            refs = refs.unsqueeze(1)  # ensure (B, R, C, H, W)
        B, R, C, H, W = refs.shape
        flat = refs.reshape(B * R, C, H, W)
        feats = self.net(flat).reshape(B, R, self.embed_dim)
        if ref_valid is None:
            return feats.mean(dim=1)
        mask = ref_valid.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (feats * mask).sum(dim=1) / denom


class ContentEncoder(nn.Module):
    """Lightweight CNN producing a global content vector from the content image.

    The content image is the source-glyph render (1 ch in Stage A; cached
    content fields with more channels in Stage B/C).
    """

    def __init__(self, in_channels: int, embed_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embed_dim = embed_dim

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        return self.net(content).flatten(1)


# --------------------------------------------------------------------------- #
# Top-level config + model                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class ConditioningBundle:
    """Container for all non-time conditioning signals.

    Bundling these means ``encode_conditioning`` / ``predict_logits_*`` /
    ``sample_states`` take one dataclass instead of a long argument list. The
    timestep tensor stays a positional arg because it is the diffusion
    variable (changes every step), whereas everything in this bundle is
    typically constructed once per batch.

    All fields default to ``None`` so callers only fill the channels they need;
    the model fills any missing id with its learned null embedding (for
    classifier-free guidance) and skips the style encoder if ``ref_images`` is
    absent. ``style_family_id`` and ``unit_id`` are accepted for cross-paper
    signature compatibility and currently ignored by QT-Font.
    """

    content: torch.Tensor
    char_id: torch.Tensor | None = None
    script_id: torch.Tensor | None = None
    writer_id: torch.Tensor | None = None
    style_family_id: torch.Tensor | None = None
    unit_id: torch.Tensor | None = None
    ref_images: torch.Tensor | None = None
    ref_valid: torch.Tensor | None = None


@dataclass
class QTFontConfig:
    """Configuration for QTFontModel.

    Most numeric fields are guesses; see ``reports/blind_impl.md`` for citations.
    """

    image_size: int = 128
    in_channels: int = 1
    content_channels: int = 1
    ref_channels: int = 1

    depth: int = 4  # leaf grid = 2^depth × 2^depth (16×16 default)
    n_states: int = 8  # categorical bins per leaf

    hidden_dim: int = 128
    n_layers: int = 3
    time_embed_dim: int = 128
    style_embed_dim: int = 128
    content_embed_dim: int = 128

    char_vocab_size: int = 64
    writer_vocab_size: int = 24
    script_vocab_size: int = 5

    dropout: float = 0.0
    ref_dropout: float = 0.1

    timesteps: int = 100

    # Aux fields the synthetic batch may carry but we ignore.
    extras: dict = field(default_factory=dict)


class QTFontModel(nn.Module):
    """Dual quadtree graph U-Net for discrete diffusion in leaf state space."""

    def __init__(self, cfg: QTFontConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._build_graph_buffers(cfg)
        self._build_conditioning_modules(cfg)
        self._build_graph_modules(cfg)

    # ------------------------------------------------------------------ #
    # __init__ helpers (split for readability and unit-testability)       #
    # ------------------------------------------------------------------ #

    def _build_graph_buffers(self, cfg: QTFontConfig) -> None:
        """Topology buffers: parent/child indices, neighbour grids, leaf maps.

        Pure index gymnastics — no learnable parameters. All registered as
        non-persistent buffers so they ride along on ``.to(device)`` calls
        but stay out of ``state_dict``.
        """
        parent_of, child_of = _build_parent_child_index(cfg.depth)
        leaf_neigh, leaf_xy = _build_leaf_sibling_index(cfg.depth, cfg.image_size)
        self.register_buffer("parent_of", parent_of, persistent=False)
        self.register_buffer("child_of", child_of, persistent=False)
        self.register_buffer("leaf_neigh", leaf_neigh, persistent=False)
        self.register_buffer("leaf_xy", leaf_xy, persistent=False)

        # Leaves = last level; parents = penultimate level (coarse graph).
        level_start, level_size = _full_quadtree_offsets(cfg.depth)
        leaf_start = level_start[-1]
        leaf_end = leaf_start + level_size[-1]
        parent_start = level_start[-2]
        parent_end = parent_start + level_size[-2]
        self.leaf_start = leaf_start
        self.leaf_end = leaf_end
        self.n_leaves = level_size[-1]
        self.n_parents = level_size[-2]

        # Per-parent child indices in leaf-local coords (subtract leaf_start).
        parent_children = child_of[parent_start:parent_end].clone()  # (P, 4) global ids
        mask = parent_children >= 0
        parent_children_local = parent_children.clone()
        parent_children_local[mask] = parent_children[mask] - leaf_start
        self.register_buffer("parent_children_local", parent_children_local, persistent=False)

        # Inverse map: leaf id → parent id within the coarse level.
        leaf_to_parent = torch.zeros(level_size[-1], dtype=torch.long)
        for p in range(level_size[-2]):
            for k in range(_QUAD):
                lid = int(parent_children_local[p, k].item())
                if 0 <= lid < level_size[-1]:
                    leaf_to_parent[lid] = p
        self.register_buffer("leaf_to_parent", leaf_to_parent, persistent=False)

        # Coarse-graph neighbours: parents live on a 2^(depth-1) grid.
        coarse_neigh, _coarse_xy = _build_leaf_sibling_index(cfg.depth - 1, cfg.image_size)
        self.register_buffer("coarse_neigh", coarse_neigh, persistent=False)

    def _build_conditioning_modules(self, cfg: QTFontConfig) -> None:
        """Conditioning encoders + embedding tables (time, content, style, ids)."""
        # State + 2D positional embeddings for leaves.
        self.state_embed = nn.Embedding(cfg.n_states, cfg.hidden_dim)
        grid = 2**cfg.depth
        self.row_embed = nn.Embedding(grid, cfg.hidden_dim)
        self.col_embed = nn.Embedding(grid, cfg.hidden_dim)

        # Time / content / style.
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.content_encoder = ContentEncoder(cfg.content_channels, cfg.content_embed_dim)
        self.content_proj = nn.Linear(cfg.content_embed_dim, cfg.hidden_dim)
        self.style_encoder = StyleEncoder(cfg.ref_channels, cfg.style_embed_dim)
        self.style_proj = nn.Linear(cfg.style_embed_dim, cfg.hidden_dim)

        # Categorical embeddings (+1 row each for the CFG null id).
        self.char_embed = nn.Embedding(cfg.char_vocab_size + 1, cfg.hidden_dim)
        self.writer_embed = nn.Embedding(cfg.writer_vocab_size + 1, cfg.hidden_dim)
        self.script_embed = nn.Embedding(cfg.script_vocab_size + 1, cfg.hidden_dim)
        self.null_char_id = cfg.char_vocab_size
        self.null_writer_id = cfg.writer_vocab_size
        self.null_script_id = cfg.script_vocab_size

    def _build_graph_modules(self, cfg: QTFontConfig) -> None:
        """Dual quadtree graph stacks, content-aware pool, broadcast, and head."""
        self.fine_layers = nn.ModuleList(
            [GraphConv(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.n_layers)]
        )
        self.coarse_layers = nn.ModuleList(
            [GraphConv(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.n_layers)]
        )
        self.pool = ContentAwarePool(cfg.hidden_dim)
        # Coarse → fine broadcast through a small MLP.
        self.parent_to_child = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        # Output head: per-leaf state logits.
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.n_states),
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.ref_dropout_p = cfg.ref_dropout

    # ------------------------------------------------------------------ #
    # forward                                                             #
    # ------------------------------------------------------------------ #

    def encode_conditioning(
        self,
        timesteps: torch.Tensor,
        cond_bundle: ConditioningBundle,
    ) -> torch.Tensor:
        """Fuse (timesteps, content, style, ids) into a single (B, hidden) vector.

        Missing categorical ids fall back to the learned null-id row to keep the
        classifier-free guidance path consistent. ``style_family_id`` and
        ``unit_id`` are part of :class:`ConditioningBundle` for cross-paper
        compatibility but are ignored here.
        """
        cfg = self.cfg
        time_in = _timestep_embedding(timesteps, cfg.time_embed_dim)
        cond = self.time_mlp(time_in)

        cond = cond + self.content_proj(self.content_encoder(cond_bundle.content))

        if cond_bundle.ref_images is not None:
            style_vec = self.style_encoder(cond_bundle.ref_images, cond_bundle.ref_valid)
            cond = cond + self.style_proj(style_vec)

        cond = cond + self._embed_id_with_null(
            cond_bundle.char_id, self.char_embed, self.null_char_id, cond
        )
        cond = cond + self._embed_id_with_null(
            cond_bundle.writer_id, self.writer_embed, self.null_writer_id, cond
        )
        cond = cond + self._embed_id_with_null(
            cond_bundle.script_id, self.script_embed, self.null_script_id, cond
        )
        return cond  # (B, hidden)

    @staticmethod
    def _embed_id_with_null(
        ids: torch.Tensor | None,
        embed: nn.Embedding,
        null_id: int,
        ref_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Embed ``ids``, or the all-null row if ``ids is None``.

        ``ref_tensor`` is only used to source ``batch_size`` and ``device`` when
        synthesising the null id tensor.
        """
        if ids is not None:
            return embed(ids)
        null = torch.full(
            (ref_tensor.shape[0],), null_id, dtype=torch.long, device=ref_tensor.device
        )
        return embed(null)

    def predict_state_logits(
        self,
        leaf_states: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Run the dual-graph U-Net on (B, L) leaf states + (B, hidden) cond.

        Returns per-leaf state logits (B, L, K).
        """
        h = self.state_embed(leaf_states)  # (B, L, C)
        # Add 2D positional embeddings.
        row = self.leaf_xy[:, 0]
        col = self.leaf_xy[:, 1]
        h = h + self.row_embed(row).unsqueeze(0) + self.col_embed(col).unsqueeze(0)
        # Inject conditioning broadcast.
        h = h + cond.unsqueeze(1)

        # Fine graph stack.
        for layer in self.fine_layers:
            h = layer(h, self.leaf_neigh)
            h = self.dropout(h)

        # Pool fine → coarse via content-aware pool.
        coarse = self.pool(h, self.parent_children_local)  # (B, P, C)
        # Add conditioning at coarse too.
        coarse = coarse + cond.unsqueeze(1)

        # Coarse graph stack.
        for layer in self.coarse_layers:
            coarse = layer(coarse, self.coarse_neigh)
            coarse = self.dropout(coarse)

        # Broadcast coarse back to fine (each leaf receives its parent feature).
        broadcast = self.parent_to_child(coarse)[:, self.leaf_to_parent, :]  # (B, L, C)
        h = h + broadcast

        return self.head(h)  # (B, L, K)

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        content: torch.Tensor,
        char_id: torch.Tensor | None = None,
        writer_id: torch.Tensor | None = None,
        script_id: torch.Tensor | None = None,
        style_family_id: torch.Tensor | None = None,
        unit_id: torch.Tensor | None = None,
        ref_images: torch.Tensor | None = None,
        ref_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard pixel-in / pixel-out adapter.

        Internally converts ``x_t`` (B,1,H,W) into per-leaf categorical states,
        runs the dual quadtree graph denoiser, decodes back to a pixel image.
        This makes the model drop-in for the shared GaussianDiffusion sampler
        and the smoke-test harness, even though the *native* training loss
        operates on discrete state logits (see :func:`qt_font.train.compute_loss`).

        The broad keyword surface is preserved here for cross-paper signature
        compatibility (shared smoke harness calls every model with the same
        kwargs); internally we collapse it into a :class:`ConditioningBundle`
        and delegate. ``style_family_id`` and ``unit_id`` are stored on the
        bundle but currently ignored by QT-Font.
        """
        bundle = ConditioningBundle(
            content=content,
            char_id=char_id,
            script_id=script_id,
            writer_id=writer_id,
            style_family_id=style_family_id,
            unit_id=unit_id,
            ref_images=ref_images,
            ref_valid=ref_valid,
        )
        cfg = self.cfg
        leaf_states = quantize_to_states(x_t, depth=cfg.depth, n_states=cfg.n_states)
        cond = self.encode_conditioning(timesteps, bundle)
        logits = self.predict_state_logits(leaf_states, cond)
        return decode_states_to_image(
            logits, depth=cfg.depth, n_states=cfg.n_states, image_size=cfg.image_size
        )

    # Used by train.compute_loss to get raw logits (no decode) for D3PM CE loss.
    def predict_logits_from_image(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        cond_bundle: ConditioningBundle,
    ) -> torch.Tensor:
        cfg = self.cfg
        leaf_states = quantize_to_states(x_t, depth=cfg.depth, n_states=cfg.n_states)
        cond = self.encode_conditioning(timesteps, cond_bundle)
        return self.predict_state_logits(leaf_states, cond)

    def predict_logits_from_states(
        self,
        leaf_states: torch.Tensor,
        timesteps: torch.Tensor,
        cond_bundle: ConditioningBundle,
    ) -> torch.Tensor:
        cond = self.encode_conditioning(timesteps, cond_bundle)
        return self.predict_state_logits(leaf_states, cond)


def build_qt_font(cfg: QTFontConfig) -> QTFontModel:
    return QTFontModel(cfg)
