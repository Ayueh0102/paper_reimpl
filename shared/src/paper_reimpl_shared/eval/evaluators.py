"""Evaluator policy helpers.

The generator must never report fake evaluator metrics. This module only
validates whether frozen evaluator checkpoints are available and returns a
machine-readable status for reports/training summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_EVALUATORS = ("R_char", "R_writer", "R_style_family", "R_script")


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base / path


def resolve_evaluator_policy(config: dict[str, Any] | None, *, base: str | Path) -> dict[str, Any]:
    """Validate and summarize evaluator availability.

    Supported modes:
    - placeholder: evaluators are explicitly unavailable and unused.
    - frozen_ckpt: every required evaluator path must exist.
    """

    cfg = dict(config or {})
    mode = str(cfg.get("mode", "placeholder")).strip().lower()
    ckpts = cfg.get("ckpts", {}) or {}
    if not isinstance(ckpts, dict):
        raise ValueError("evaluator_policy.ckpts must be a mapping")

    base_path = Path(base)
    evaluators: dict[str, dict[str, Any]] = {}
    if mode == "placeholder":
        for name in REQUIRED_EVALUATORS:
            value = ckpts.get(name)
            evaluators[name] = {
                "status": "unavailable",
                "path": str(value) if value else "",
                "used_in_loss": False,
                "used_for_metrics": False,
            }
        return {
            "mode": "placeholder",
            "status": "unavailable",
            "used_in_loss": False,
            "used_for_metrics": False,
            "evaluators": evaluators,
            "note": "No evaluator ckpts configured; report metrics as unavailable.",
        }

    if mode == "frozen_ckpt":
        missing: list[str] = []
        for name in REQUIRED_EVALUATORS:
            value = ckpts.get(name)
            if not value:
                missing.append(name)
                evaluators[name] = {
                    "status": "missing",
                    "path": "",
                    "used_in_loss": False,
                    "used_for_metrics": False,
                }
                continue
            path = _resolve_path(str(value), base=base_path)
            exists = path.exists()
            if not exists:
                missing.append(name)
            evaluators[name] = {
                "status": "available" if exists else "missing",
                "path": str(path),
                "used_in_loss": exists,
                "used_for_metrics": exists,
            }
        if missing:
            raise FileNotFoundError(f"Missing frozen evaluator ckpts: {', '.join(missing)}")
        return {
            "mode": "frozen_ckpt",
            "status": "available",
            "used_in_loss": True,
            "used_for_metrics": True,
            "evaluators": evaluators,
            "note": "Frozen evaluator ckpts validated.",
        }

    raise ValueError(f"Unsupported evaluator_policy.mode: {mode}")
