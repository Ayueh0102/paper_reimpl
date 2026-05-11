"""fontdiffuser blind reimplementation (Phase 1).

Public surface kept minimal — every consumer should import from the named
sub-modules to make the dependency graph obvious during code review.
"""

from .model import (
    FontDiffuser,
    FontDiffuserConfig,
    StyleExtractor,
    build_fontdiffuser,
)
from .train import compute_loss, main as train_main
from .sample import sample, sample_ddim, sample_ddpm

__all__ = [
    "FontDiffuser",
    "FontDiffuserConfig",
    "StyleExtractor",
    "build_fontdiffuser",
    "compute_loss",
    "sample",
    "sample_ddim",
    "sample_ddpm",
    "train_main",
]
