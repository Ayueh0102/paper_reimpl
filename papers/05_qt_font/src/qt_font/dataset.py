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


class _TTFPairAdapter(torch.utils.data.Dataset):
    """Wrap TTFCrossFontPairDataset to match the legacy `refs` batch key
    that QT-Font's model.compute_loss and train.compute_loss read.

    Shared TTF dataset emits `ref_images` (a list of [1, C, H, W] tensors).
    QT's train loop expects either `refs` (already stacked) or `ref_images`.
    We stack to `refs` here so DataLoader's default collate produces a
    single [B, N, C, H, W] tensor without needing a custom collate_fn.
    """

    def __init__(self, inner) -> None:
        super().__init__()
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.inner[idx]
        refs = s["ref_images"]
        refs_t = torch.stack(refs, dim=0) if isinstance(refs, list) else refs
        return {
            "image": s["image"],
            "content": s["content"],
            "refs": refs_t,
            "char_id": torch.as_tensor(s["char_id"], dtype=torch.long),
            "writer_id": torch.as_tensor(s["writer_id"], dtype=torch.long),
            "script_id": torch.as_tensor(s["script_id"], dtype=torch.long),
        }


def build_ttf_dataset(
    *,
    fonts_root,
    image_size: int,
    content_channels: int,
    n_refs: int = 1,
    font_size_ratio: float = 0.85,
    length: int = 10_000,
    seed: int = 42,
    cjk_start: int = 0x4E00,
    cjk_end: int = 0x9FFF,
    char_cache_path=None,
    font_ids: list[str] | None = None,
    script_categories: dict | None = None,
) -> _TTFPairAdapter:
    """Real cross-font TTF dataset for QT-Font Stage A."""
    from pathlib import Path
    from paper_reimpl_shared.data.ttf_pair_dataset import TTFCrossFontPairDataset

    fonts_root = Path(fonts_root)
    if char_cache_path is None:
        char_cache_path = fonts_root / f".ttf_supported_chars_{image_size}px_{font_size_ratio}.json"
    inner = TTFCrossFontPairDataset(
        fonts_root=fonts_root,
        font_ids=font_ids,
        image_size=image_size,
        content_channels=content_channels,
        font_size_ratio=font_size_ratio,
        length=length,
        ref_count=n_refs,
        seed=seed,
        ensure_diff_source=True,
        cjk_start=cjk_start,
        cjk_end=cjk_end,
        char_cache_path=char_cache_path,
        script_categories=script_categories,
    )
    return _TTFPairAdapter(inner)


class _ManifestAdapter(Dataset):
    """Adapt :class:`CalligraphyJsonlDataset` to the schema QT-Font expects.

    Legacy dataset emits ``ref_images`` (list of [1, H, W] tensors). 05's
    train loop reads ``refs`` (stacked [N, 1, H, W]) plus the categorical ids.
    """

    def __init__(self, inner) -> None:
        super().__init__()
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.inner[idx]
        refs = s["ref_images"]
        if isinstance(refs, list):
            refs_t = torch.stack(refs, dim=0) if refs else torch.zeros(0, 1, self.inner.image_size, self.inner.image_size)
        else:
            refs_t = refs
        return {
            "image": s["image"],
            "content": s["content"],
            "refs": refs_t,
            "char_id": torch.as_tensor(s["char_id"], dtype=torch.long),
            "writer_id": torch.as_tensor(s["writer_id"], dtype=torch.long),
            "script_id": torch.as_tensor(s["script_id"], dtype=torch.long),
        }


def build_manifest_dataset(
    *,
    manifest_path,
    image_size: int,
    content_channels,
    max_refs: int = 0,
) -> _ManifestAdapter:
    """Real manifest-backed Stage B/C dataset for QT-Font."""
    from paper_reimpl_shared.data.legacy import CalligraphyJsonlDataset

    inner = CalligraphyJsonlDataset(
        manifest_path,
        image_size=image_size,
        content_channels=list(content_channels) if not isinstance(content_channels, list) else content_channels,
        max_refs=max_refs,
    )
    return _ManifestAdapter(inner)
