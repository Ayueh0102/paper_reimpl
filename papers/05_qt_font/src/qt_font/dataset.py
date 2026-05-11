"""Tiny dataset adapter for QT-Font.

For Phase 1 we keep the dataset thin: a synthetic generator that mimics the
``make_synthetic_batch`` keys, plus a placeholder ``ManifestDataset`` slot
that consumes the shared manifest path resolver.

The shared smoke harness drives the model end-to-end with synthetic data, so
this module exists mostly to keep the public API symmetric with the other
papers and to give Stage A/B/C config a place to land.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from paper_reimpl_shared.runner.smoke import make_synthetic_batch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


@dataclass
class SyntheticConfig:
    """Synthetic dataset config (smoke / dry-run)."""

    length: int = 64
    image_size: int = 64
    in_channels: int = 1
    content_channels: int = 1
    n_refs: int = 1
    char_vocab_size: int = 64
    writer_vocab_size: int = 24
    script_vocab_size: int = 5
    seed: int = 0


class SyntheticDataset(IterableDataset):
    """Streams ``make_synthetic_batch``-shaped batches of size 1.

    Wrapped by the standard DataLoader (batch_size > 1) to assemble batches.
    """

    def __init__(self, cfg: SyntheticConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # Shard work across DataLoader workers so that ``num_workers > 0`` does
        # not silently N-fold the epoch (each worker would otherwise run the
        # full iterator independently).
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
            # Derive a per-sample seed so that runs with the same
            # ``SyntheticConfig.seed`` produce identical streams (and so each
            # worker draws distinct samples).
            sample_seed = self.cfg.seed + worker_id * self.cfg.length + i
            batch = make_synthetic_batch(
                batch_size=1,
                image_size=self.cfg.image_size,
                in_channels=self.cfg.in_channels,
                char_vocab_size=self.cfg.char_vocab_size,
                writer_vocab_size=self.cfg.writer_vocab_size,
                n_refs=self.cfg.n_refs,
                device="cpu",
                seed=sample_seed,
            )
            # The shared smoke generator yields content with 3 channels; we may
            # downcast to ``content_channels`` if the model wants fewer (eg. 1
            # for Stage A bitmap-only). Index along channel dim.
            if batch["content"].shape[1] != self.cfg.content_channels:
                batch["content"] = batch["content"][:, : self.cfg.content_channels]
            # Squeeze batch dimension so DataLoader can collate properly.
            yield {k: v.squeeze(0) for k, v in batch.items()}


def build_dataset(cfg: SyntheticConfig) -> SyntheticDataset:
    """Factory mirroring other papers' ``build_dataset`` API."""
    return SyntheticDataset(cfg)


class ManifestPlaceholder(Dataset):
    """Stub for the manifest-backed Stage A/B/C dataset.

    Phase 1 sandbox doesn't need real data plumbing — the shared entrypoint
    accepts ``--synthetic`` which routes to :class:`SyntheticDataset`. Phase 2/3
    will fill this in with actual manifest reads via
    ``paper_reimpl_shared.data.manifest``.
    """

    def __init__(self, *_, **__) -> None:
        super().__init__()
        raise NotImplementedError(
            "ManifestPlaceholder not implemented yet — use SyntheticDataset for "
            "Phase 1 dry-runs and smoke tests."
        )

    def __len__(self) -> int:  # pragma: no cover
        return 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError
