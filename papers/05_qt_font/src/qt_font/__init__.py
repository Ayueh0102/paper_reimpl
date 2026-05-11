"""qt_font — Phase 2 paper-aligned reimplementation of QT-Font (SIGGRAPH 2024).

Public exports:
    ConditioningBundle    : back-compat bundle (most fields are ignored — the
                            paper-aligned model takes octrees, not categorical
                            ids; the kwargs are kept so existing call sites
                            don't break).
    QTFontConfig          : Phase 2 hyper-parameters (depth=7, K=3, T=1000, ...).
    QTFontModel           : graph-U-Net + multi-depth split prediction model.
    build_qt_font         : factory matching the other papers' `build_<paper>`.
    D3PMUniform           : K=3 cosine-schedule D3PM with full Q + Q_cum buffers.

    OctreeBatch           : the sparse octree container.
    extract_glyph_labels  : pixel image → {bg, contour, skeleton} label map.
    build_octree_from_labels / build_octree_from_image : build the sparse octree.
    render_label_image    : OctreeBatch → dense (B, H, W) label image.

    compute_multi_depth_ce : training loss; per-depth split CE + leaf 3-class CE.
"""

from __future__ import annotations

from .losses import compute_multi_depth_ce
from .model import (
    ConditioningBundle,
    D3PMUniform,
    QTFontConfig,
    QTFontModel,
    build_qt_font,
    render_label_image,
)
from .octree import (
    OctreeBatch,
    QuadTreeLevel,
    build_octree_from_image,
    build_octree_from_labels,
    extract_glyph_labels,
)

__all__ = [
    "ConditioningBundle",
    "D3PMUniform",
    "OctreeBatch",
    "QTFontConfig",
    "QTFontModel",
    "QuadTreeLevel",
    "build_octree_from_image",
    "build_octree_from_labels",
    "build_qt_font",
    "compute_multi_depth_ce",
    "extract_glyph_labels",
    "render_label_image",
]
