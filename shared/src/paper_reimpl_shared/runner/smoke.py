"""Synthetic batch generators for smoke tests.

Each paper's `tests/test_smoke.py` should use these to verify:
  1. model forward without crash
  2. model backward + 1 optimizer step
  3. sampler produces correct-shape output

No disk I/O. CPU-only by default.
"""

from __future__ import annotations

import torch


def make_synthetic_batch(
    *,
    batch_size: int = 2,
    image_size: int = 128,
    in_channels: int = 1,
    char_vocab_size: int = 100,
    writer_vocab_size: int = 24,
    n_refs: int = 4,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Standard synthetic batch dictionary used by all papers' smoke tests."""
    return {
        "image": torch.randn(batch_size, in_channels, image_size, image_size, device=device),
        "content": torch.randn(batch_size, 3, image_size, image_size, device=device),
        "refs": torch.randn(batch_size, n_refs, in_channels, image_size, image_size, device=device),
        "char_id": torch.randint(0, char_vocab_size, (batch_size,), device=device),
        "writer_id": torch.randint(0, writer_vocab_size, (batch_size,), device=device),
        "script_id": torch.randint(0, 5, (batch_size,), device=device),  # 楷/行/草/隸/篆
    }
