"""Content cache loader (bitmap / SDF / skeleton npz channels).

Lifted from mother repo's data.py with stable signature so paper subclasses
can compose without touching legacy. NPZ files are stored per-character
under ``<root>/<height>/<unicode_hex>.npz``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch


CHANNEL_NAMES = ("bitmap", "sdf", "skeleton", "edge", "distance_transform", "stroke_axis")


def load_content_npz(
    path: str | Path, *, channels: Sequence[str], image_size: int
) -> torch.Tensor:
    """Load named channels from a content cache .npz file.

    Args:
        path: .npz file path.
        channels: ordered list of channel names to stack along channel axis.
        image_size: expected (H, W); fails if mismatch.

    Returns:
        Tensor [C, H, W] float32 in [-1, 1].
    """
    arr = np.load(path)
    stack: list[np.ndarray] = []
    for ch in channels:
        if ch not in arr:
            raise KeyError(f"Channel '{ch}' missing in {path}; available={list(arr.keys())}")
        plane = arr[ch].astype(np.float32)
        if plane.shape != (image_size, image_size):
            raise ValueError(
                f"Channel '{ch}' shape {plane.shape} != ({image_size},{image_size}) in {path}"
            )
        stack.append(plane)
    out = np.stack(stack, axis=0)
    return torch.from_numpy(out)


def synthetic_content(channels: Sequence[str], image_size: int) -> torch.Tensor:
    """Random content for smoke tests (no disk I/O)."""
    return torch.zeros(len(channels), image_size, image_size)
