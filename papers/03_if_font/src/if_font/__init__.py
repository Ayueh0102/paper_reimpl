"""IF-Font reimplementation — IDS-conditioned VQ autoregressive token decoder.

Paper: IF-Font, NeurIPS 2024 (Stareven233/IF-Font).
Public API (Phase 2): ``IFFont``, ``IFFontConfig``, ``build_if_font``,
``IDSTokenizer``, ``IDSResolver``, ``VQTokenizerAdapter``,
``StyleEncoder``, ``MoCoWrapper``.
"""

from __future__ import annotations

from . import losses
from .ids import (
    DEFAULT_IDC_CHARS,
    IDSResolver,
    IDSTokenizer,
    parse_structure_class,
)
from .model import (
    IDSEmbedding,
    IFFont,
    IFFontConfig,
    MoCoWrapper,
    StyleEncoder,
    TransformerARDecoder,
    VQTokenizer,
    VQTokenizerAdapter,
    VQTokenizerConfig,
    build_if_font,
)
from .train import MoCoCache, compute_loss

__all__ = [
    "DEFAULT_IDC_CHARS",
    "IDSEmbedding",
    "IDSResolver",
    "IDSTokenizer",
    "IFFont",
    "IFFontConfig",
    "MoCoCache",
    "MoCoWrapper",
    "StyleEncoder",
    "TransformerARDecoder",
    "VQTokenizer",
    "VQTokenizerAdapter",
    "VQTokenizerConfig",
    "build_if_font",
    "compute_loss",
    "losses",
    "parse_structure_class",
]
