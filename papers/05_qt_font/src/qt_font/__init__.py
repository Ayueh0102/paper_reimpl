"""qt_font — blind reimplementation of QT-Font (SIGGRAPH 2024).

Public exports:
    ConditioningBundle    : dataclass bundling all non-time conditioning signals
                            (content, char_id, script_id, writer_id, refs, ...)
    QTFontConfig          : dataclass describing model hyper-parameters
    QTFontModel           : the full quadtree-graph discrete-diffusion model
    build_qt_font         : factory matching the other papers' `build_<paper>`
    D3PMUniform           : uniform discrete diffusion utility (forward q_sample
                            + loss); ``nn.Module`` so it moves with `.to(device)`.
    build_quadtree_states : differentiable-free helper turning a pixel grid into
                            per-leaf categorical states for a full quadtree.
    quantize_to_states    : pixel image → per-leaf categorical state indices.
    decode_states_to_image: per-leaf softmax over K classes → pixel image.
"""

from __future__ import annotations

from .model import (
    ConditioningBundle,
    D3PMUniform,
    QTFontConfig,
    QTFontModel,
    build_qt_font,
    build_quadtree_states,
    decode_states_to_image,
    quantize_to_states,
)

__all__ = [
    "ConditioningBundle",
    "D3PMUniform",
    "QTFontConfig",
    "QTFontModel",
    "build_qt_font",
    "build_quadtree_states",
    "decode_states_to_image",
    "quantize_to_states",
]
