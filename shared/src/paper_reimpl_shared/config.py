"""Small config helpers used by Experiment A scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return data


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing mapping config section: {key}")
    return value


def require_positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Config value must be a positive int: {key}")
    return value


def resolve_path(value: str | Path, *, base: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or base is None:
        return path
    return Path(base).expanduser() / path
