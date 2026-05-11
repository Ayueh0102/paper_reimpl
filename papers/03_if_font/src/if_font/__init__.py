"""IF-Font blind reimplementation — IDS-conditioned VQ autoregressive token decoder.

Paper: IF-Font, NeurIPS 2024.
Public API: ``IFFont``, ``IFFontConfig``, ``build_if_font``, ``IDSTokenizer``.
"""

from __future__ import annotations

from .ids import (
    DEFAULT_IDC_CHARS,
    IDSTokenizer,
    parse_structure_class,
)
from .model import (
    IFFont,
    IFFontConfig,
    TransformerARDecoder,
    VQTokenizer,
    VQTokenizerConfig,
    build_if_font,
)
from .train import compute_loss

__all__ = [
    "DEFAULT_IDC_CHARS",
    "IDSTokenizer",
    "IFFont",
    "IFFontConfig",
    "TransformerARDecoder",
    "VQTokenizer",
    "VQTokenizerConfig",
    "build_if_font",
    "compute_loss",
    "parse_structure_class",
]
