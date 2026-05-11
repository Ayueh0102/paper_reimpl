"""Datasets and batch helpers for Experiment A training."""

from __future__ import annotations

import io
import json
import random
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


SCRIPT_CLASS_ORDER = ["楷書", "行書", "草書", "小篆"]

# Per-worker in-memory caches. After the first successful read, subsequent
# reads of the same file hit RAM (no E:\\ I/O). Workers don't share memory, so
# each of the 4 dataloader workers builds its own cache (~200 MB images + ~1 GB
# npz arrays per worker). Empirically the cache fully warms within ~2 epochs.
_image_bytes_cache: dict[str, bytes] = {}
_npz_arrays_cache: dict[str, dict[str, np.ndarray]] = {}


def _read_bytes_with_retry(path: str | Path, *, attempts: int = 6, base_delay: float = 0.1) -> bytes:
    """open + read all bytes with backoff retry. Used so we can cache successful
    reads (and so PIL can reopen from BytesIO if Image.open's lazy-read fails on
    a flaky drive)."""
    key = str(path)
    cached = _image_bytes_cache.get(key)
    if cached is not None:
        return cached
    last: Exception | None = None
    for i in range(attempts):
        try:
            with open(path, "rb") as f:
                data = f.read()
            _image_bytes_cache[key] = data
            return data
        except OSError as exc:
            last = exc
            time.sleep(base_delay * (2 ** i))
    raise last  # type: ignore[misc]


def _open_with_retry(path: str | Path, *, attempts: int = 6, base_delay: float = 0.1) -> Image.Image:
    """PIL Image.open via cached bytes — first call reads E:\\ with retry then
    caches; later calls open from BytesIO (zero E:\\ I/O)."""
    return Image.open(io.BytesIO(_read_bytes_with_retry(path, attempts=attempts, base_delay=base_delay)))


def load_grayscale_tensor(path: str | Path, *, image_size: int) -> torch.Tensor:
    image = _open_with_retry(path).convert("L")
    image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (image_size, image_size), color=255)
    left = (image_size - image.width) // 2
    top = (image_size - image.height) // 2
    canvas.paste(image, (left, top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None] * 2.0 - 1.0


def load_content_tensor(path: str | Path, *, channels: list[str], image_size: int) -> torch.Tensor:
    key = str(path)
    arrays = _npz_arrays_cache.get(key)
    if arrays is None:
        last: Exception | None = None
        for i in range(6):
            try:
                with np.load(path) as raw:
                    # Only cache numeric channel arrays — npz can also store string
                    # metadata keys (e.g. script_label) that fail astype(float32).
                    arrays = {}
                    for ch in channels:
                        if ch not in raw:
                            continue
                        arrays[ch] = np.asarray(raw[ch], dtype=np.float32)
                _npz_arrays_cache[key] = arrays
                break
            except (OSError, zipfile.BadZipFile, EOFError, ValueError) as exc:
                # Flaky NVMe can return half-written or zero-length npz; numpy
                # raises BadZipFile / ValueError("not a zip file") in those
                # cases, not OSError, so retry on the broader set.
                last = exc
                time.sleep(0.1 * (2 ** i))
        else:
            raise last  # type: ignore[misc]
    data = arrays
    tensors: list[torch.Tensor] = []
    for channel in channels:
        if channel not in data:
            raise KeyError(f"Missing content channel `{channel}` in {path}")
        arr = np.asarray(data[channel], dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Content channel must be HxW: {channel} in {path}")
        tensor = torch.from_numpy(arr)[None]
        if tensor.shape[-2:] != (image_size, image_size):
            tensor = F.interpolate(tensor[None], size=(image_size, image_size), mode="bilinear", align_corners=False)[0]
        tensors.append(tensor.clamp(-1.0, 1.0))
    return torch.cat(tensors, dim=0)


def _id_for(value: str, mapping: dict[str, int]) -> int:
    if value not in mapping:
        mapping[value] = len(mapping)
    return mapping[value]


class CalligraphyJsonlDataset(Dataset):
    """JSONL manifest dataset.

    Expected fields follow `experiments/A_unit_geometry_jit/PLAN.md` §4.4.
    References can be omitted for A0/A1. If present, use `ref_image_paths`.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        image_size: int,
        content_channels: list[str],
        max_refs: int = 0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.content_channels = content_channels
        self.max_refs = max_refs
        self.rows: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"Manifest has no rows: {self.manifest_path}")

        self.writer_to_id: dict[str, int] = {}
        self.style_family_to_id: dict[str, int] = {}
        self.script_to_id = {label: i for i, label in enumerate(SCRIPT_CLASS_ORDER)}
        self.has_manifest_label_ids = all(
            "writer_label_id" in row and ("style_family_label_id" in row or "unit_label_id" in row)
            for row in self.rows
        )
        self.has_manifest_char_label_ids = all("char_label_id" in row for row in self.rows)
        if self.has_manifest_label_ids:
            self.writer_vocab_size = max(
                max(int(row["writer_label_id"]) for row in self.rows) + 1,
                max(int(row.get("writer_vocab_size", 0)) for row in self.rows),
            )
            self.style_family_vocab_size = max(
                max(int(row.get("style_family_label_id", row.get("unit_label_id", 0))) for row in self.rows) + 1,
                max(int(row.get("style_family_vocab_size", row.get("unit_vocab_size", 0))) for row in self.rows),
            )
        else:
            for row in self.rows:
                _id_for(str(row.get("writer", "")), self.writer_to_id)
                _id_for(str(row.get("style_family_id", row.get("style_unit_id", ""))), self.style_family_to_id)
            self.writer_vocab_size = max(1, len(self.writer_to_id))
            self.style_family_vocab_size = max(1, len(self.style_family_to_id))
        self.unit_vocab_size = self.style_family_vocab_size
        self.char_to_id: dict[str, int] = {}
        if self.has_manifest_char_label_ids:
            self.char_vocab_size = max(
                max(int(row["char_label_id"]) for row in self.rows) + 1,
                max(int(row.get("char_vocab_size", 0)) for row in self.rows),
            )
        else:
            for row in self.rows:
                _id_for(str(row.get("char", row.get("target_char", row.get("char_id", "")))), self.char_to_id)
            self.char_vocab_size = max(1, len(self.char_to_id))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Outer retry: if any file load OSErrors after its own internal retries
        # (typical case: flaky storage like the E:\ drive on this machine),
        # resample a different index instead of crashing the whole training.
        # Caps at 8 attempts so a permanently-broken row doesn't loop forever.
        last_err: Exception | None = None
        for _ in range(8):
            try:
                return self._fetch(index)
            except OSError as exc:
                last_err = exc
                index = random.randrange(len(self.rows))
        raise last_err  # type: ignore[misc]

    def _fetch(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = load_grayscale_tensor(row["image_path"], image_size=self.image_size)
        content = load_content_tensor(row["content_npz"], channels=self.content_channels, image_size=self.image_size)
        ref_paths = list(row.get("ref_image_paths", []))[: self.max_refs]
        refs = [load_grayscale_tensor(path, image_size=self.image_size) for path in ref_paths]
        style_family_id = (
            int(row.get("style_family_label_id", row.get("unit_label_id", 0)))
            if self.has_manifest_label_ids
            else self.style_family_to_id[str(row.get("style_family_id", row.get("style_unit_id", "")))]
        )
        char_label_id = (
            int(row["char_label_id"])
            if self.has_manifest_char_label_ids
            else self.char_to_id[str(row.get("char", row.get("target_char", row.get("char_id", ""))))]
        )
        return {
            "image": image,
            "content": content,
            "char_id": char_label_id,
            "script_id": int(row.get("script_label_id", self.script_to_id.get(str(row.get("script", "")), 0))),
            "writer_id": int(row["writer_label_id"]) if self.has_manifest_label_ids else self.writer_to_id[str(row.get("writer", ""))],
            "style_family_id": style_family_id,
            "unit_id": style_family_id,
            "ref_images": refs,
            "metadata": row,
        }

    def metadata(self) -> dict[str, int]:
        return {
            "writer_vocab_size": self.writer_vocab_size,
            "style_family_vocab_size": self.style_family_vocab_size,
            "unit_vocab_size": self.style_family_vocab_size,
            "char_vocab_size": self.char_vocab_size,
        }


class SyntheticCalligraphyDataset(Dataset):
    """Tiny synthetic dataset for model-build and 1-step smoke validation."""

    def __init__(
        self,
        *,
        length: int,
        image_size: int,
        content_channels: int,
        writer_vocab_size: int = 4,
        style_family_vocab_size: int = 8,
        unit_vocab_size: int | None = None,
        char_vocab_size: int = 64,
        script_vocab_size: int = 3,
        ref_count: int = 0,
        seed: int = 42,
    ) -> None:
        self.length = length
        self.image_size = image_size
        self.content_channels = content_channels
        self.writer_vocab_size = writer_vocab_size
        self.style_family_vocab_size = int(style_family_vocab_size if unit_vocab_size is None else unit_vocab_size)
        self.char_vocab_size = char_vocab_size
        self.script_vocab_size = script_vocab_size
        self.ref_count = ref_count
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.length

    def _glyph_like_tensor(self, index: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(index)
        base = torch.ones(1, self.image_size, self.image_size)
        yy = torch.linspace(-1.0, 1.0, self.image_size).view(1, self.image_size, 1)
        xx = torch.linspace(-1.0, 1.0, self.image_size).view(1, 1, self.image_size)
        for _ in range(4):
            angle = torch.rand((), generator=g) * torch.pi
            offset = torch.rand((), generator=g) * 1.2 - 0.6
            width = torch.rand((), generator=g) * 0.04 + 0.025
            line = torch.abs(torch.cos(angle) * xx + torch.sin(angle) * yy - offset) < width
            base = torch.where(line, torch.full_like(base, -1.0), base)
        noise = torch.randn(base.shape, generator=g) * 0.03
        return (base + noise).clamp(-1.0, 1.0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = self._glyph_like_tensor(index)
        content = image.repeat(self.content_channels, 1, 1) if self.content_channels else torch.empty(0)
        if self.content_channels:
            content = content + torch.randn_like(content) * 0.02
            content = content.clamp(-1.0, 1.0)
        refs = [self._glyph_like_tensor(index * 100 + i + 1) for i in range(self.ref_count)]
        return {
            "image": image,
            "content": content,
            "char_id": index % max(1, self.char_vocab_size),
            "script_id": index % max(1, self.script_vocab_size),
            "writer_id": index % max(1, self.writer_vocab_size),
            "style_family_id": index % max(1, self.style_family_vocab_size),
            "unit_id": index % max(1, self.style_family_vocab_size),
            "ref_images": refs,
            "metadata": {"synthetic_index": index},
        }

    def metadata(self) -> dict[str, int]:
        return {
            "writer_vocab_size": self.writer_vocab_size,
            "style_family_vocab_size": self.style_family_vocab_size,
            "unit_vocab_size": self.style_family_vocab_size,
            "char_vocab_size": self.char_vocab_size,
        }


def collate_calligraphy_batch(batch: list[dict[str, Any]], *, max_refs: int = 0) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch])
    content = torch.stack([item["content"] for item in batch])
    char_id = torch.tensor([item.get("char_id", 0) for item in batch], dtype=torch.long)
    script_id = torch.tensor([item["script_id"] for item in batch], dtype=torch.long)
    writer_id = torch.tensor([item["writer_id"] for item in batch], dtype=torch.long)
    style_family_id = torch.tensor(
        [item["style_family_id"] if "style_family_id" in item else item["unit_id"] for item in batch],
        dtype=torch.long,
    )

    ref_tensors: list[torch.Tensor] = []
    ref_valid_rows: list[torch.Tensor] = []
    for item in batch:
        refs = list(item.get("ref_images", []))[:max_refs]
        n_real = len(refs)
        while len(refs) < max_refs:
            refs.append(torch.zeros_like(images[0]))
        ref_tensors.append(torch.stack(refs) if max_refs else torch.empty(0, *images.shape[1:]))
        # ref_valid: True at positions with a real ref, False at zero-padded slots.
        # Consumed by RefTokenEncoder so padded positions become a learnable
        # [PAD] embedding instead of patches of literal zeros.
        valid = torch.zeros(max_refs, dtype=torch.bool)
        if n_real > 0:
            valid[:n_real] = True
        ref_valid_rows.append(valid)

    ref_images = torch.stack(ref_tensors) if max_refs else torch.empty(images.shape[0], 0, *images.shape[1:])
    ref_valid = (
        torch.stack(ref_valid_rows) if max_refs else torch.empty(images.shape[0], 0, dtype=torch.bool)
    )
    return {
        "image": images,
        "content": content,
        "char_id": char_id,
        "script_id": script_id,
        "writer_id": writer_id,
        "style_family_id": style_family_id,
        "unit_id": style_family_id,
        "ref_images": ref_images,
        "ref_valid": ref_valid,
        "metadata": [item["metadata"] for item in batch],
    }
