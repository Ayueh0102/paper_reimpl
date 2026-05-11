"""hfh_font — blind Phase-1 reimplementation of HFH-Font (SIGGRAPH-Asia 2024).

Public surface:
- :class:`hfh_font.model.HFHFontModel` — top-level model
- :class:`hfh_font.model.ModelConfig` — dataclass mirror of ``configs/model.yaml``
- :func:`hfh_font.model.build_model`
- :func:`hfh_font.train.main` — orchestrator entry, called by
  ``paper_reimpl_shared.runner.entrypoint``
- :mod:`hfh_font.sample` — inference helpers
"""

from __future__ import annotations

from .model import (
    HFHFontModel,
    LatentUNet,
    ModelConfig,
    StyleGuidedSR,
    TinyVAE,
    build_model,
)

__all__ = [
    "HFHFontModel",
    "LatentUNet",
    "ModelConfig",
    "StyleGuidedSR",
    "TinyVAE",
    "build_model",
]
