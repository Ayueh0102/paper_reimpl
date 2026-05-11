"""qt_font — blind reimplementation of QT-Font (SIGGRAPH 2024).

Public exports:
    QTFontConfig    : dataclass describing model hyper-parameters
    QTFontModel     : the full quadtree-graph discrete-diffusion model
    build_qt_font   : factory matching the other papers' `build_<paper>` convention
    D3PMUniform     : uniform discrete diffusion utility (forward q_sample + loss)
    build_quadtree_states : differentiable-free helper turning a pixel grid into
                            per-leaf categorical states for a full quadtree.
"""

from __future__ import annotations

from .model import (
    D3PMUniform,
    QTFontConfig,
    QTFontModel,
    build_qt_font,
    build_quadtree_states,
    decode_states_to_image,
    quantize_to_states,
)

__all__ = [
    "D3PMUniform",
    "QTFontConfig",
    "QTFontModel",
    "build_qt_font",
    "build_quadtree_states",
    "decode_states_to_image",
    "quantize_to_states",
]
