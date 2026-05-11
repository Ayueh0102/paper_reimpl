"""Blind reimplementation of Calliffusion (Liao 2023, arXiv 2305.19124).

Modules
-------
- ``model``  : conditional DDPM U-Net with text cross-attention at every stage.
- ``text``   : Chinese BERT wrapper (with offline stub fallback for smoke tests).
- ``lora``   : minimal LoRA wrapper for ``nn.Linear``.
- ``dataset``: text-prompt calligraphy dataset built on top of the shared
  ``CalligraphyJsonlDataset``.
- ``train``  : training entry called by ``paper_reimpl_shared.runner.entrypoint``.
- ``sample`` : DDPM/DDIM sampling helper.
"""

from __future__ import annotations

__all__ = [
    "model",
    "text",
    "lora",
    "dataset",
    "train",
    "sample",
]
