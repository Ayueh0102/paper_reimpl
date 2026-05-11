"""Unified training/eval entry point.

CLI surface (frozen):
    --paper        paper short name (fontdiffuser/hfh_font/...)
    --data         path to data_*.yaml
    --model        path to model.yaml
    --train        path to train_*.yaml
    --manifest     manifest file name (overrides data.yaml)
    --device       cuda:0 / cpu
    --init-ckpt    warm-start checkpoint
    --resume       resume from latest in ckpt_dir
    --dry-run      validate config + load 1 batch + 1 forward, then exit
    --synthetic    use random tensors instead of real data
    --data-backend {mac_symlink, lab_server, vast_snapshot}

This file is the single integration point. Per-paper trainers are picked up
by ``--paper`` and dispatched to ``papers.<paper>.train.main``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from paper_reimpl_shared.config import load_yaml
from paper_reimpl_shared.data.manifest import Backend, BackendPaths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="paper_reimpl unified entry point")
    p.add_argument("--paper", required=True, help="paper short name (fontdiffuser/hfh_font/...)")
    p.add_argument("--data", required=True, help="data YAML path")
    p.add_argument("--model", required=True, help="model YAML path")
    p.add_argument("--train", required=True, help="train YAML path")
    p.add_argument("--manifest", default=None, help="manifest file name override")
    p.add_argument("--device", default="cuda:0", help="cuda:0 / cuda:1 / cpu")
    p.add_argument("--init-ckpt", default=None, help="warm-start checkpoint")
    p.add_argument("--resume", action="store_true", help="resume from latest in ckpt_dir")
    p.add_argument("--dry-run", action="store_true", help="validate + 1-batch forward, then exit")
    p.add_argument("--synthetic", action="store_true", help="random tensors instead of real data")
    p.add_argument(
        "--data-backend",
        choices=["mac_symlink", "lab_server", "vast_snapshot"],
        default="mac_symlink",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = BackendPaths.resolve(args.data_backend)
    print(f"[entrypoint] paper={args.paper} backend={args.data_backend}")
    print(f"[entrypoint] manifest_root={paths.manifest_root}")
    print(f"[entrypoint] device={args.device} dry_run={args.dry_run} synthetic={args.synthetic}")

    # Load configs (validates YAML parseability)
    data_cfg = load_yaml(args.data)
    model_cfg = load_yaml(args.model)
    train_cfg = load_yaml(args.train)

    # Each paper exposes a `train.main(args, data_cfg, model_cfg, train_cfg, paths)` entry.
    # Convention: papers/<NN>_<short>/src/<short>/train.py
    paper_module = f"{args.paper}.train"
    try:
        mod = importlib.import_module(paper_module)
    except ModuleNotFoundError as e:
        print(f"[entrypoint] ERROR: cannot import {paper_module}: {e}")
        print(f"[entrypoint] Did you `uv pip install -e .` in papers/<NN>_{args.paper}/?")
        return 1

    return mod.main(args, data_cfg=data_cfg, model_cfg=model_cfg, train_cfg=train_cfg, paths=paths)


if __name__ == "__main__":
    sys.exit(main())
