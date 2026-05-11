"""Minimal sparse 2D octree (a.k.a. quadtree) for QT-Font.

Why this file exists
--------------------
QT-Font's reference implementation (lsflyt-pku/QT-Font) is built on a 2-D port
of OCNN/DualOctreeGNN (`ocnn_2d/`). That stack is a sizeable C++/CUDA library
of its own; pulling it in as a dependency would dwarf the rest of this repo
and make CPU-only smoke tests slow. Instead, we implement just the pieces the
QT-Font paper actually uses on top of pure PyTorch:

* **Adaptive sparse subdivision**. Top ``full_depth`` levels are dense (every
  node exists). Beyond ``full_depth``, a node only exists if its parent has at
  least one non-empty grand-descendant. This is the same regime as
  ``ocnn_2d.Octree.build_octree`` (``third_party/05_qt_font/ocnn_2d/octree/octree.py:161``).
* **Per-level node lists** keyed by ``(batch_id, row, col)``. We do not bother
  with Morton-key bit packing — the depths we work at (≤ 7 for 128 px) keep
  the per-level tensor sizes small enough that explicit (row, col) indexing
  is fine, and it is much easier to read.
* **Multi-depth supervision**. At every inner depth we know which nodes are
  ``split`` (have any non-empty descendant) vs. ``empty`` — this is the label
  the paper's `axis_loss` predicts at every inner level
  (``third_party/05_qt_font/losses/loss.py:28``). At the leaf depth we attach
  a 3-class label ``{0=bg, 1=contour, 2=skeleton}``.

Edge-direction-aware graph adjacency
------------------------------------
``modules_bn.GraphConv`` in the official repo takes 5 edge types
(N/E/S/W/self) and one weight matrix per ``node_type`` (per depth). We mirror
that here: each per-depth :class:`QuadTreeLevel` exposes an ``edge_index`` over
its sparse node list with the corresponding ``edge_type``. The 5th edge type
("self") is implicit and added as a residual inside the GraphConv module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# 3-class label extraction (paper data_pp delta D1, D2, D6).                   #
# --------------------------------------------------------------------------- #


def extract_glyph_labels(image_uint8: np.ndarray) -> np.ndarray:
    """Image → ``{0=bg, 1=contour, 2=skeleton}`` label map (H, W) int8.

    Mirrors ``third_party/05_qt_font/datasets/chinesefont_asymmetric.py:303-309``.

    Parameters
    ----------
    image_uint8 : numpy array (H, W), dtype uint8.
        Single-channel grayscale glyph. The paper assumes white-paper / black-ink
        glyphs and binarises ``> 127.5``; we then **invert** so the ink side is
        the foreground, and run contour + skeleton extraction on that.
    """
    import cv2 as cv
    from skimage import morphology

    if image_uint8.ndim != 2:
        raise ValueError(f"expected (H, W) uint8 image, got {image_uint8.shape}")

    # Paper convention: white-paper / black-ink → binarise, then invert so the
    # ink is "foreground" (255). Without this step `findContours` would return
    # contours of the paper boundary, not the glyph.
    binary = (image_uint8 > 127).astype(np.uint8)  # 0=ink, 1=paper
    fg = (1 - binary) * 255  # 255=ink, 0=paper

    # Skeletonise the binary foreground.
    skeleton = morphology.skeletonize(fg // 255).astype(np.uint8)
    # Contour as a 1-px line on a blank canvas.
    contours, _hierarchy = cv.findContours(fg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros_like(fg, dtype=np.uint8)
    if len(contours) > 0:
        cv.drawContours(canvas, contours, -1, color=1, thickness=1)
    label = canvas + skeleton * 2
    # Where contour and skeleton coincide, contour wins (matches the paper
    # convention `label[label == 3] = 1` in dataset.py:118).
    label[label > 2] = 1
    return label.astype(np.int8)


# --------------------------------------------------------------------------- #
# Sparse octree data structures.                                               #
# --------------------------------------------------------------------------- #


@dataclass
class QuadTreeLevel:
    """One depth of the sparse octree.

    Fields
    ------
    depth : int
        Octree depth ``d``. Grid side at this depth is ``2**d``.
    xy : LongTensor (N, 2)
        ``(row, col)`` of each node on the ``2**d × 2**d`` grid.
    batch_id : LongTensor (N,)
        Which sample in the batch this node belongs to.
    parent : LongTensor (N,)
        Index into level ``d-1``'s node list (or ``-1`` for ``d == full_depth``).
    edge_index : LongTensor (2, E)
        ``(src, dst)`` over the sparse node list. Source/dest are local to this
        level. Self-loops are NOT included; the GraphConv adds them implicitly.
    edge_type : LongTensor (E,)
        Direction code per edge: ``0=N``, ``1=E``, ``2=S``, ``3=W``. (Mirrors
        the official ``edge_dir`` in ``third_party/05_qt_font/models/dual_octree.py``.)
    split : LongTensor (N,)
        Inner-level supervision: ``1`` if this node has ≥1 non-empty descendant
        at depth ``d+1``, ``0`` otherwise. At the deepest level the field is
        all zeros — supervision there uses :attr:`leaf_label` instead.
    leaf_label : LongTensor (N,) | None
        Defined only at the leaf depth: 3-class state ``{0=bg, 1=contour, 2=skeleton}``.
        ``None`` at inner depths.
    """

    depth: int
    xy: torch.Tensor
    batch_id: torch.Tensor
    parent: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    split: torch.Tensor
    leaf_label: torch.Tensor | None = None

    def __len__(self) -> int:
        return int(self.xy.shape[0])


@dataclass
class OctreeBatch:
    """A batch of sparse octrees, level-by-level.

    ``levels[d]`` is the :class:`QuadTreeLevel` for depth ``d``. The container
    is contiguous from ``full_depth`` down to ``depth`` (lower indices are not
    materialised — they are dense by definition and never participate in the
    sparse loss).
    """

    batch_size: int
    full_depth: int
    depth: int
    levels: dict[int, QuadTreeLevel]

    def to(self, device: torch.device | str) -> OctreeBatch:
        new_levels: dict[int, QuadTreeLevel] = {}
        for d, lvl in self.levels.items():
            new_levels[d] = QuadTreeLevel(
                depth=lvl.depth,
                xy=lvl.xy.to(device),
                batch_id=lvl.batch_id.to(device),
                parent=lvl.parent.to(device),
                edge_index=lvl.edge_index.to(device),
                edge_type=lvl.edge_type.to(device),
                split=lvl.split.to(device),
                leaf_label=None if lvl.leaf_label is None else lvl.leaf_label.to(device),
            )
        return OctreeBatch(
            batch_size=self.batch_size,
            full_depth=self.full_depth,
            depth=self.depth,
            levels=new_levels,
        )


# --------------------------------------------------------------------------- #
# Build helpers.                                                               #
# --------------------------------------------------------------------------- #


def _build_edges_4conn_per_batch(
    xy: torch.Tensor, batch_id: torch.Tensor, B: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """4-connectivity edges within each batch element.

    Edge type codes mirror the official 4-direction enumeration:
        0 = North (row -1), 1 = East (col +1), 2 = South (row +1), 3 = West (col -1)

    Iterating per-batch keeps the lookup dict small (one batch at a time) and
    rules out spurious cross-batch edges in one shot. At the depths we work at
    (≤ 7 → ≤ 16k nodes/batch) Python-side iteration is fine; CUDA-graph-style
    batched ops are a Phase 4 optimisation if profile shows it matters.
    """
    src_list: list[int] = []
    dst_list: list[int] = []
    type_list: list[int] = []

    offsets = [(-1, 0, 0), (0, 1, 1), (1, 0, 2), (0, -1, 3)]
    xy_list = xy.tolist()
    bid_list = batch_id.tolist()

    # Build a key → global_idx map, with the batch_id baked in so look-ups are
    # batch-local. The arithmetic ``b * 1e12 + r * 1e6 + c`` is plenty of head
    # room for any depth we'd run.
    key_to_idx: dict[int, int] = {}
    for i, ((r, c), b) in enumerate(zip(xy_list, bid_list, strict=True)):
        key_to_idx[b * 1_000_000_000_000 + r * 1_000_000 + c] = i

    for i, ((r, c), b) in enumerate(zip(xy_list, bid_list, strict=True)):
        for dr, dc, etype in offsets:
            nr, nc = r + dr, c + dc
            key = b * 1_000_000_000_000 + nr * 1_000_000 + nc
            j = key_to_idx.get(key, -1)
            if j >= 0:
                src_list.append(i)
                dst_list.append(j)
                type_list.append(etype)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long).reshape(2, -1)
    edge_type = torch.tensor(type_list, dtype=torch.long)
    return edge_index, edge_type


def _build_one_sample(label: np.ndarray, full_depth: int, depth: int) -> dict[int, dict]:
    """Build the per-sample sparse octree (no batch IDs, no edges yet)."""
    H, W = label.shape
    grid_full = 1 << depth
    assert H == grid_full and W == grid_full, (
        f"label shape {label.shape} != ({grid_full}, {grid_full})"
    )

    # Per-depth occupancy mask. Build from leaf depth up: a parent is "split"
    # iff any of its 4 children at the next depth contain a non-empty cell.
    masks: dict[int, np.ndarray] = {}
    leaf_mask = label != 0  # True wherever the cell carries either contour or skeleton.
    masks[depth] = leaf_mask
    for d in range(depth - 1, full_depth - 1, -1):
        side = 1 << d
        m_child = masks[d + 1]
        m = m_child.reshape(side, 2, side, 2).any(axis=(1, 3))
        masks[d] = m

    levels: dict[int, dict] = {}
    # full_depth is dense — every position is a node, regardless of occupancy.
    side_full = 1 << full_depth
    rs, cs = np.meshgrid(np.arange(side_full), np.arange(side_full), indexing="ij")
    xy_full = np.stack([rs.reshape(-1), cs.reshape(-1)], axis=-1)
    split_full = masks[full_depth].reshape(-1).astype(np.int64)
    levels[full_depth] = {
        "xy": torch.from_numpy(xy_full).long(),
        "split": torch.from_numpy(split_full),
        "parent_xy": torch.full((side_full * side_full, 2), -1, dtype=torch.long),
        "leaf_label": None,
    }

    # Sparse levels: only keep positions whose parent is split.
    for d in range(full_depth + 1, depth + 1):
        side = 1 << d
        parent_split_grid = masks[d - 1]  # (parent_side, parent_side)
        # Expand parent split into the finer (4 children) layout.
        parent_split_fine = np.repeat(np.repeat(parent_split_grid, 2, axis=0), 2, axis=1)
        keep = parent_split_fine.reshape(side, side)
        rs, cs = np.where(keep)
        xy = np.stack([rs, cs], axis=-1)
        parent_xy = np.stack([rs // 2, cs // 2], axis=-1)
        if d < depth:
            split_d = masks[d][rs, cs].astype(np.int64)
        else:
            split_d = np.zeros_like(rs, dtype=np.int64)
        levels[d] = {
            "xy": torch.from_numpy(xy).long(),
            "split": torch.from_numpy(split_d),
            "parent_xy": torch.from_numpy(parent_xy).long(),
            "leaf_label": (
                torch.from_numpy(label[rs, cs]).long() if d == depth else None
            ),
        }
    return levels


def build_octree_from_labels(
    labels: torch.Tensor, *, full_depth: int, depth: int
) -> OctreeBatch:
    """Build a batched sparse octree from a stack of 3-class label maps.

    Parameters
    ----------
    labels : LongTensor (B, H, W)
        Per-pixel ``{0=bg, 1=contour, 2=skeleton}`` labels. ``H == W == 2**depth``.
    full_depth : int
        Depths ``[0, full_depth]`` are kept dense.
    depth : int
        Maximum depth (leaf depth).
    """
    if labels.dim() != 3:
        raise ValueError(f"expected (B, H, W) labels, got shape {tuple(labels.shape)}")
    B = labels.shape[0]
    grid = 1 << depth
    if labels.shape[1] != grid or labels.shape[2] != grid:
        raise ValueError(
            f"labels grid {tuple(labels.shape[1:])} != ({grid}, {grid}) for depth={depth}"
        )

    # Build each sample separately.
    per_sample: list[dict[int, dict]] = []
    labels_np = labels.cpu().numpy().astype(np.int64)
    for b in range(B):
        per_sample.append(_build_one_sample(labels_np[b], full_depth, depth))

    # Concatenate per-level across the batch. The interesting bit is mapping
    # the per-sample ``parent_xy`` to a global index into level d-1's
    # concatenated node list. We pre-build a per-sample (xy → local_idx)
    # lookup and a per-sample running-offset table for each prior depth.
    local_index_per_depth: dict[int, list[dict[int, int]]] = {}
    offsets_per_depth: dict[int, list[int]] = {}
    levels_out: dict[int, QuadTreeLevel] = {}

    for d in range(full_depth, depth + 1):
        # Compute offsets table for this level (used by level d+1).
        offsets = []
        acc = 0
        for b in range(B):
            offsets.append(acc)
            acc += per_sample[b][d]["xy"].shape[0]
        offsets_per_depth[d] = offsets

        # Per-sample (xy_key → local_idx) lookup for this level.
        local_maps: list[dict[int, int]] = []
        side_d = 1 << d
        for b in range(B):
            xy_b = per_sample[b][d]["xy"]
            local: dict[int, int] = {}
            for li, (r, c) in enumerate(xy_b.tolist()):
                local[r * side_d + c] = li
            local_maps.append(local)
        local_index_per_depth[d] = local_maps

        # Concatenate the per-sample tensors with batch_id and globalised parent.
        xy_chunks: list[torch.Tensor] = []
        split_chunks: list[torch.Tensor] = []
        batch_chunks: list[torch.Tensor] = []
        parent_chunks: list[torch.Tensor] = []
        leaf_chunks: list[torch.Tensor] = []

        for b in range(B):
            lvl = per_sample[b][d]
            n = lvl["xy"].shape[0]
            xy_chunks.append(lvl["xy"])
            split_chunks.append(lvl["split"])
            batch_chunks.append(torch.full((n,), b, dtype=torch.long))

            if d == full_depth:
                parent_chunks.append(torch.full((n,), -1, dtype=torch.long))
            else:
                parent_xy = lvl["parent_xy"]
                pside = 1 << (d - 1)
                parent_local_map = local_index_per_depth[d - 1][b]
                parent_offset = offsets_per_depth[d - 1][b]
                keys = (parent_xy[:, 0] * pside + parent_xy[:, 1]).tolist()
                # By construction every parent must exist (we only include a
                # node at depth d if its parent at d-1 is "split"). Raise on
                # mismatch — it would mean the build invariant is broken.
                try:
                    p_local = torch.tensor(
                        [parent_local_map[k] for k in keys], dtype=torch.long
                    )
                except KeyError as e:
                    raise RuntimeError(
                        f"parent key {e.args[0]} not found at depth {d-1} for sample {b}"
                    ) from None
                parent_chunks.append(p_local + parent_offset)

            leaf_lbl = lvl["leaf_label"]
            if leaf_lbl is not None:
                leaf_chunks.append(leaf_lbl)

        xy_cat = torch.cat(xy_chunks, dim=0) if xy_chunks else torch.zeros((0, 2), dtype=torch.long)
        batch_cat = (
            torch.cat(batch_chunks, dim=0) if batch_chunks else torch.zeros((0,), dtype=torch.long)
        )
        split_cat = (
            torch.cat(split_chunks, dim=0) if split_chunks else torch.zeros((0,), dtype=torch.long)
        )
        parent_cat = (
            torch.cat(parent_chunks, dim=0) if parent_chunks else torch.zeros((0,), dtype=torch.long)
        )
        leaf_cat = torch.cat(leaf_chunks, dim=0) if leaf_chunks else None

        # 4-connectivity edges within each batch element.
        edge_index, edge_type = _build_edges_4conn_per_batch(xy_cat, batch_cat, B)

        levels_out[d] = QuadTreeLevel(
            depth=d,
            xy=xy_cat,
            batch_id=batch_cat,
            parent=parent_cat,
            edge_index=edge_index,
            edge_type=edge_type,
            split=split_cat,
            leaf_label=leaf_cat,
        )

    return OctreeBatch(
        batch_size=B,
        full_depth=full_depth,
        depth=depth,
        levels=levels_out,
    )


def render_label_image(
    octree: OctreeBatch,
    *,
    use_leaf_label: bool = True,
) -> torch.Tensor:
    """Render the leaf-depth labels back into a dense (B, H, W) image.

    Useful for visualisation and for the sampler's "next-step input" loop in
    ``sample.py`` (`p_sample` in ``third_party/05_qt_font/main.py:258``).

    Missing leaves are filled with 0 (background).
    """
    leaf = octree.levels[octree.depth]
    B = octree.batch_size
    side = 1 << octree.depth
    img = torch.zeros((B, side, side), dtype=torch.long, device=leaf.xy.device)
    if leaf.leaf_label is None or not use_leaf_label:
        # Render binary occupancy.
        img[leaf.batch_id, leaf.xy[:, 0], leaf.xy[:, 1]] = 1
    else:
        img[leaf.batch_id, leaf.xy[:, 0], leaf.xy[:, 1]] = leaf.leaf_label
    return img


def build_octree_from_image(
    image: torch.Tensor, *, full_depth: int, depth: int
) -> OctreeBatch:
    """Convenience: glyph image → 3-class labels → sparse octree.

    Parameters
    ----------
    image : Tensor (B, 1, H, W) in [-1, 1] OR (B, H, W) uint8 in [0, 255]
        Glyph batch. Float tensors are converted to uint8 via ``((x+1)/2*255)``.
    """
    if image.dim() == 4:
        if image.shape[1] != 1:
            raise ValueError(f"expected 1-channel image, got shape {tuple(image.shape)}")
        image = image.squeeze(1)
    if image.dtype.is_floating_point:
        image_u8 = ((image.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    else:
        image_u8 = image.to(torch.uint8)
    image_np = image_u8.cpu().numpy()
    labels_np = np.stack([extract_glyph_labels(image_np[b]) for b in range(image_np.shape[0])])
    labels = torch.from_numpy(labels_np.astype(np.int64))
    return build_octree_from_labels(labels, full_depth=full_depth, depth=depth)


__all__ = [
    "QuadTreeLevel",
    "OctreeBatch",
    "extract_glyph_labels",
    "build_octree_from_labels",
    "build_octree_from_image",
    "render_label_image",
]
