"""Backend-aware manifest path resolver.

Selects path prefix at runtime so the same YAML config works on Mac (via
mother_repo_link symlink), on the lab server (via scp-ed data_snapshot),
and on vast.ai (via /workspace/data_snapshot).

CLI flag: ``--data-backend {mac_symlink, lab_server, vast_snapshot}``.

Each paper's data_<stage>.yaml only stores manifest *file names* and split
identifiers (relative paths). The backend rewrites the root prefix at load
time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Backend = Literal["mac_symlink", "lab_server", "vast_snapshot"]


@dataclass(frozen=True)
class BackendPaths:
    """Resolved root paths for a given backend."""

    manifest_root: Path
    content_cache_root: Path
    ttf_root: Path

    @classmethod
    def resolve(cls, backend: Backend, *, repo_root: Path | None = None) -> "BackendPaths":
        """Return root paths for the requested backend.

        Args:
            backend: one of mac_symlink / lab_server / vast_snapshot.
            repo_root: paper_reimpl repo root. Defaults to current cwd's
                ancestor containing ``mother_repo_link``.
        """
        if backend == "mac_symlink":
            root = repo_root or _find_repo_root()
            link = root / "mother_repo_link"
            if not link.exists():
                raise FileNotFoundError(
                    f"mother_repo_link missing at {link}; create symlink to mother repo."
                )
            mother = link.resolve()
            return cls(
                manifest_root=mother / "experiments" / "A_unit_geometry_jit" / "outputs" / "splits",
                content_cache_root=mother
                / "experiments"
                / "A_unit_geometry_jit"
                / "outputs"
                / "content_fields_cache",
                ttf_root=mother / "data" / "ttf_renders",
            )

        if backend == "lab_server":
            base = Path(
                os.environ.get("PR_DATA_ROOT", r"D:\Char\ayueh\paper_reimpl\data_snapshot")
            )
            return cls(
                manifest_root=base / "splits",
                content_cache_root=base / "content_fields_cache",
                ttf_root=base / "ttf_renders",
            )

        if backend == "vast_snapshot":
            base = Path(os.environ.get("PR_DATA_ROOT", "/workspace/data_snapshot"))
            return cls(
                manifest_root=base / "splits",
                content_cache_root=base / "content_fields_cache",
                ttf_root=base / "ttf_renders",
            )

        raise ValueError(f"Unknown backend: {backend}")


def _find_repo_root() -> Path:
    """Walk upward from cwd looking for ``mother_repo_link`` or `.git`."""
    here = Path.cwd().resolve()
    for parent in (here, *here.parents):
        if (parent / "mother_repo_link").exists() or (parent / ".git").is_dir():
            return parent
    raise FileNotFoundError("Cannot find paper_reimpl repo root from cwd.")


def manifest_path(manifest_name: str, *, backend: Backend, repo_root: Path | None = None) -> Path:
    """Resolve a manifest file name to an absolute path."""
    paths = BackendPaths.resolve(backend, repo_root=repo_root)
    candidate = paths.manifest_root / manifest_name
    if not candidate.exists():
        raise FileNotFoundError(f"Manifest not found: {candidate} (backend={backend})")
    return candidate


def content_cache_path(
    unicode_hex: str, *, height: int, backend: Backend, repo_root: Path | None = None
) -> Path:
    """Resolve content cache npz path by unicode hex + cached height."""
    paths = BackendPaths.resolve(backend, repo_root=repo_root)
    candidate = paths.content_cache_root / str(height) / f"{unicode_hex}.npz"
    return candidate


def ttf_root(*, backend: Backend, repo_root: Path | None = None) -> Path:
    return BackendPaths.resolve(backend, repo_root=repo_root).ttf_root
