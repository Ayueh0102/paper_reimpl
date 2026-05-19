"""Manifest-backed DataLoader sampling helpers.

The shared ``CalligraphyJsonlDataset`` exposes raw JSONL rows as ``rows``.
These helpers keep writer-balancing policy out of paper-specific training
loops while remaining no-op for synthetic / TTF datasets.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

import torch
from torch.utils.data import Sampler, SubsetRandomSampler, WeightedRandomSampler

__all__ = [
    "build_manifest_train_sampler",
    "manifest_writer_labels",
]


def _writer_key(row: dict[str, Any]) -> str:
    for key in ("writer_id", "writer", "writer_label_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def manifest_writer_labels(dataset: Any) -> list[str] | None:
    """Return per-row writer labels for manifest datasets, else ``None``."""
    rows = getattr(dataset, "rows", None)
    if not isinstance(rows, list) or not rows:
        return None
    if not all(isinstance(row, dict) for row in rows):
        return None
    return [_writer_key(row) for row in rows]


def _sampling_cfg(data_cfg: dict[str, Any], train_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for source in (data_cfg, train_cfg):
        nested = source.get("sampling")
        if isinstance(nested, dict):
            cfg.update(nested)
    for source in (data_cfg, train_cfg):
        for key in (
            "writer_balanced_sampling",
            "writer_max_samples_per_epoch",
            "max_samples_per_writer_per_epoch",
            "samples_per_epoch",
        ):
            if key in source:
                cfg[key] = source[key]
    return cfg


def _mode_from_cfg(cfg: dict[str, Any]) -> str:
    mode = str(cfg.get("mode", "")).strip().lower()
    if not mode and bool(cfg.get("writer_balanced_sampling", False)):
        mode = "writer_balanced"
    if not mode and (
        cfg.get("writer_max_samples_per_epoch") is not None
        or cfg.get("max_samples_per_writer_per_epoch") is not None
    ):
        mode = "writer_cap"
    return mode or "none"


def _cap_sampler(
    labels: list[str],
    *,
    max_per_writer: int,
    samples_per_epoch: int | None,
    seed: int,
) -> SubsetRandomSampler[int]:
    by_writer: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_writer[label].append(idx)

    rng = random.Random(seed)
    selected: list[int] = []
    for label in sorted(by_writer):
        indices = list(by_writer[label])
        rng.shuffle(indices)
        selected.extend(indices[:max_per_writer])
    rng.shuffle(selected)
    if samples_per_epoch is not None:
        selected = selected[:samples_per_epoch]

    generator = torch.Generator()
    generator.manual_seed(seed)
    return SubsetRandomSampler(selected, generator=generator)


def _balanced_sampler(
    labels: list[str],
    *,
    samples_per_epoch: int | None,
    replacement: bool,
    seed: int,
) -> WeightedRandomSampler:
    counts = Counter(labels)
    weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights,
        num_samples=int(samples_per_epoch or len(labels)),
        replacement=replacement,
        generator=generator,
    )


def build_manifest_train_sampler(
    dataset: Any,
    *,
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    seed: int,
) -> Sampler[int] | None:
    """Build an optional manifest sampler from config.

    Supported config forms, either in data YAML or train YAML:

    ``sampling: {mode: writer_balanced, samples_per_epoch: 50000}``
        WeightedRandomSampler with inverse writer-frequency weights.

    ``sampling: {mode: writer_cap, max_samples_per_writer_per_epoch: 2000}``
        Randomly caps each writer's contribution to an epoch.

    Legacy flat keys ``writer_balanced_sampling: true`` and
    ``writer_max_samples_per_epoch`` are also accepted.
    """
    labels = manifest_writer_labels(dataset)
    if labels is None:
        return None

    cfg = _sampling_cfg(data_cfg, train_cfg)
    mode = _mode_from_cfg(cfg)
    if mode in {"none", "false", "off"}:
        return None

    samples_per_epoch = cfg.get("samples_per_epoch")
    samples_per_epoch = int(samples_per_epoch) if samples_per_epoch is not None else None

    if mode in {"writer_balanced", "balanced", "weighted_writer"}:
        replacement = bool(cfg.get("replacement", True))
        return _balanced_sampler(
            labels,
            samples_per_epoch=samples_per_epoch,
            replacement=replacement,
            seed=seed,
        )

    if mode in {"writer_cap", "cap", "capped"}:
        max_per_writer = cfg.get(
            "max_samples_per_writer_per_epoch",
            cfg.get("writer_max_samples_per_epoch"),
        )
        if max_per_writer is None:
            raise ValueError(
                "writer_cap sampling requires max_samples_per_writer_per_epoch"
            )
        return _cap_sampler(
            labels,
            max_per_writer=int(max_per_writer),
            samples_per_epoch=samples_per_epoch,
            seed=seed,
        )

    raise ValueError(f"Unknown manifest sampling mode: {mode}")
