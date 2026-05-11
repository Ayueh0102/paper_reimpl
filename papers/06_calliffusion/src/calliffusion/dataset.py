"""Calliffusion text-prompt dataset.

Builds on top of the shared ``CalligraphyJsonlDataset`` to add the
natural-language prompt field used by Calliffusion's BERT path.

Each item produced is a dict with at least:
    image: [1, H, W] float in [-1, 1]
    prompt: str — "<char> <script> <writer>"
    char_id / script_id / writer_id: kept around for logging only.

A synthetic variant powers ``--synthetic`` smoke tests without touching disk.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from paper_reimpl_shared.data.legacy import (
    CalligraphyJsonlDataset,
    SyntheticCalligraphyDataset,
)
from torch.utils.data import Dataset


@dataclass
class PromptDatasetConfig:
    image_size: int = 64
    max_refs: int = 0
    # Token used in place of the writer / script when the field is missing.
    unk_writer: str = "[UNK_W]"
    unk_script: str = "[UNK_S]"
    # Probability that the whole prompt is dropped to an empty string. Used
    # for classifier-free guidance training. Paper does not specify; default
    # 0.1 follows common diffusion-from-text practice.
    prompt_dropout_p: float = 0.1


def _row_to_prompt(row: dict[str, Any], cfg: PromptDatasetConfig) -> str:
    char = str(row.get("char", row.get("target_char", row.get("char_id", "")))).strip()
    script = str(row.get("script", cfg.unk_script)).strip() or cfg.unk_script
    writer = str(row.get("writer", cfg.unk_writer)).strip() or cfg.unk_writer
    pieces = [p for p in (char, script, writer) if p]
    return " ".join(pieces)


class CalliffusionPromptDataset(Dataset):
    """Wrap the shared JSONL dataset and emit a ``prompt`` string per item."""

    def __init__(
        self,
        manifest_path: str,
        *,
        image_size: int = 64,
        content_channels: list[str] | None = None,
        max_refs: int = 0,
        prompt_dropout_p: float = 0.1,
    ) -> None:
        # Calliffusion does NOT use a content_field channel; we pass an empty
        # list so the shared loader skips the npz. ``CalligraphyJsonlDataset``
        # still loads the npz if any channels are requested; with an empty
        # list we hit the `if not channels` branch via the empty cat.
        content_channels = content_channels or []
        self.cfg = PromptDatasetConfig(
            image_size=image_size,
            max_refs=max_refs,
            prompt_dropout_p=prompt_dropout_p,
        )
        self._inner = CalligraphyJsonlDataset(
            manifest_path,
            image_size=image_size,
            content_channels=content_channels,
            max_refs=max_refs,
        )

    def __len__(self) -> int:
        return len(self._inner)

    def writer_names(self) -> list[str]:
        names = sorted({str(r.get("writer", "")).strip() for r in self._inner.rows})
        return [n for n in names if n]

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._inner[index]
        row = item["metadata"]
        prompt = _row_to_prompt(row, self.cfg)
        if self.cfg.prompt_dropout_p > 0 and random.random() < self.cfg.prompt_dropout_p:
            prompt = ""
        return {
            "image": item["image"],
            "prompt": prompt,
            "char_id": item["char_id"],
            "script_id": item["script_id"],
            "writer_id": item["writer_id"],
            "metadata": row,
        }


class SyntheticPromptDataset(Dataset):
    """Synthetic prompt dataset used by smoke tests + ``--synthetic`` dry-runs."""

    SCRIPTS = ["楷書", "行書", "草書", "隸書", "篆書"]

    def __init__(
        self,
        *,
        length: int = 16,
        image_size: int = 64,
        writer_vocab_size: int = 4,
        char_vocab_size: int = 16,
        prompt_dropout_p: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.length = int(length)
        self.image_size = int(image_size)
        self.writer_vocab_size = int(writer_vocab_size)
        self.char_vocab_size = int(char_vocab_size)
        self.prompt_dropout_p = float(prompt_dropout_p)
        self._inner = SyntheticCalligraphyDataset(
            length=self.length,
            image_size=self.image_size,
            content_channels=0,
            writer_vocab_size=self.writer_vocab_size,
            char_vocab_size=self.char_vocab_size,
            script_vocab_size=len(self.SCRIPTS),
            seed=seed,
        )
        # Fake-but-stable writer / char names so the prompt path exercises the
        # tokeniser. Using simple ASCII strings keeps the stub tokenizer happy
        # even when transformers is unavailable.
        self._writer_names = [f"writer{i}" for i in range(self.writer_vocab_size)]
        self._char_names = [f"char{i}" for i in range(self.char_vocab_size)]
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.length

    def writer_names(self) -> list[str]:
        return list(self._writer_names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._inner[index]
        char = self._char_names[index % self.char_vocab_size]
        script = self.SCRIPTS[index % len(self.SCRIPTS)]
        writer = self._writer_names[index % self.writer_vocab_size]
        prompt = f"{char} {script} {writer}"
        if self.prompt_dropout_p > 0 and self._rng.random() < self.prompt_dropout_p:
            prompt = ""
        return {
            "image": item["image"],
            "prompt": prompt,
            "char_id": int(item["char_id"]),
            "script_id": int(item["script_id"]),
            "writer_id": int(item["writer_id"]),
            "metadata": {"synthetic_index": index, "prompt": prompt},
        }


def collate_prompt_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch])
    prompts = [item["prompt"] for item in batch]
    char_id = torch.tensor([int(item["char_id"]) for item in batch], dtype=torch.long)
    script_id = torch.tensor([int(item["script_id"]) for item in batch], dtype=torch.long)
    writer_id = torch.tensor([int(item["writer_id"]) for item in batch], dtype=torch.long)
    return {
        "image": images,
        "prompt": prompts,
        "char_id": char_id,
        "script_id": script_id,
        "writer_id": writer_id,
        "metadata": [item["metadata"] for item in batch],
    }
