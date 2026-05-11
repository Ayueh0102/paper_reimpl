"""vq_font blind reimplementation (Phase 1).

Public surface kept minimal — every consumer should import from the named
sub-modules so the dependency graph is obvious during code review.
"""

from .dataset import (
    STRUCTURE_NAME_TO_ID,
    VQFontCollate,
    VQFontDataset,
    VQFontSyntheticDataset,
    build_dataset,
)
from .model import (
    NUM_STRUCTURE_CLASSES,
    StructureEncoder,
    StructureHead,
    TokenPriorTransformer,
    TransformerConfig,
    VQFont,
    VQFontConfig,
    VQGAN,
    VQGANConfig,
    VQGANOutputs,
    VectorQuantize,
    build_transformer,
    build_vq_font,
    build_vqgan,
)
from .sample import indices_from_logits, sample_vq_font, sample_vqgan_recon
from .train import main as train_main
from .train import transformer_compute_loss, vqgan_compute_loss

__all__ = [
    # configs / classes
    "NUM_STRUCTURE_CLASSES",
    "STRUCTURE_NAME_TO_ID",
    "StructureEncoder",
    "StructureHead",
    "TokenPriorTransformer",
    "TransformerConfig",
    "VQFont",
    "VQFontConfig",
    "VQFontCollate",
    "VQFontDataset",
    "VQFontSyntheticDataset",
    "VQGAN",
    "VQGANConfig",
    "VQGANOutputs",
    "VectorQuantize",
    # builders
    "build_dataset",
    "build_transformer",
    "build_vq_font",
    "build_vqgan",
    # samples / losses
    "indices_from_logits",
    "sample_vq_font",
    "sample_vqgan_recon",
    "transformer_compute_loss",
    "vqgan_compute_loss",
    # train entry
    "train_main",
]
