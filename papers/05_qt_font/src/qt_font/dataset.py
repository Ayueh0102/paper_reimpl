"""Tiny dataset adapter for QT-Font — Phase 2.

The shared smoke harness's ``make_synthetic_batch`` returns pure random noise
images. That is harmless for pixel-space models but **broken** for QT-Font:
``cv2.findContours`` on a near-uniform random image returns many tiny contours
that explode the sparse-octree node count and trip the multi-depth loss with
inconsistent topology between consecutive batches.

To work around that for Phase 2 smoke / dry-run we synthesise a deterministic
"checker-glyph" image with controllable contour structure — enough to exercise
the full pipeline without falling back to real data.

The :class:`ManifestPlaceholder` slot is kept for Phase 3 manifest plumbing.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


@dataclass
class SyntheticConfig:
    """Synthetic dataset config (smoke / dry-run)."""

    length: int = 64
    image_size: int = 128
    in_channels: int = 1
    content_channels: int = 1
    n_refs: int = 1
    char_vocab_size: int = 64
    writer_vocab_size: int = 24
    script_vocab_size: int = 5
    seed: int = 0


def _make_glyph_like_image(
    image_size: int, *, seed: int, in_channels: int = 1
) -> torch.Tensor:
    """Synthesise a deterministic glyph-like image with a few thick strokes.

    We draw a small number of rectangles + diagonals on a white background,
    binarise, then return as a float tensor in ``[-1, +1]``. Sufficient to
    drive contour + skeleton extraction reliably.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = torch.ones((image_size, image_size), dtype=torch.float32)  # white paper
    # 1-3 thick horizontal/vertical strokes.
    n_strokes = int(torch.randint(1, 4, (1,), generator=g).item())
    for _ in range(n_strokes):
        # Random orientation, position, thickness.
        orient = int(torch.randint(0, 2, (1,), generator=g).item())
        pos = int(torch.randint(image_size // 4, 3 * image_size // 4, (1,), generator=g).item())
        thick = int(torch.randint(2, max(3, image_size // 12), (1,), generator=g).item())
        if orient == 0:  # horizontal
            r0 = max(0, pos - thick // 2)
            r1 = min(image_size, pos + thick // 2 + 1)
            img[r0:r1, image_size // 6 : 5 * image_size // 6] = -1.0
        else:  # vertical
            c0 = max(0, pos - thick // 2)
            c1 = min(image_size, pos + thick // 2 + 1)
            img[image_size // 6 : 5 * image_size // 6, c0:c1] = -1.0
    # Replicate across channels if asked for >1.
    if in_channels == 1:
        return img.unsqueeze(0)
    return img.unsqueeze(0).expand(in_channels, -1, -1).clone()


class SyntheticDataset(IterableDataset):
    """Streams one structurally-meaningful glyph proxy per sample."""

    def __init__(self, cfg: SyntheticConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = get_worker_info()
        if info is None:
            start, end = 0, self.cfg.length
            worker_id = 0
        else:
            per_worker = int(math.ceil(self.cfg.length / info.num_workers))
            start = info.id * per_worker
            end = min(start + per_worker, self.cfg.length)
            worker_id = info.id

        for i in range(start, end):
            sample_seed = self.cfg.seed + worker_id * self.cfg.length + i
            image = _make_glyph_like_image(
                self.cfg.image_size, seed=sample_seed, in_channels=self.cfg.in_channels
            )
            content = _make_glyph_like_image(
                self.cfg.image_size,
                seed=sample_seed + 13,
                in_channels=self.cfg.content_channels,
            )
            refs = torch.stack(
                [
                    _make_glyph_like_image(
                        self.cfg.image_size,
                        seed=sample_seed + 100 + r,
                        in_channels=self.cfg.in_channels,
                    )
                    for r in range(self.cfg.n_refs)
                ],
                dim=0,
            )
            yield {
                "image": image,
                "content": content,
                "refs": refs,
                # Legacy id fields — the paper-aligned model ignores them.
                "char_id": torch.tensor(sample_seed % self.cfg.char_vocab_size, dtype=torch.long),
                "writer_id": torch.tensor(
                    sample_seed % self.cfg.writer_vocab_size, dtype=torch.long
                ),
                "script_id": torch.tensor(
                    sample_seed % self.cfg.script_vocab_size, dtype=torch.long
                ),
            }


def build_dataset(cfg: SyntheticConfig) -> SyntheticDataset:
    """Factory mirroring other papers' ``build_dataset`` API."""
    return SyntheticDataset(cfg)


class ManifestPlaceholder(Dataset):
    """Stub for the manifest-backed Stage A/B/C dataset (Phase 3)."""

    def __init__(self, *_, **__) -> None:
        super().__init__()
        raise NotImplementedError(
            "ManifestPlaceholder not implemented yet — use SyntheticDataset for "
            "Phase 2 dry-runs and smoke tests."
        )

    def __len__(self) -> int:  # pragma: no cover
        return 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError
