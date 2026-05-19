from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from paper_reimpl_shared.data.sampling import build_manifest_train_sampler

REPO_ROOT = Path(__file__).resolve().parents[2]
ENRICH_SCRIPT = REPO_ROOT / "scripts" / "enrich_manifest.py"


def _load_enrich_module():
    spec = importlib.util.spec_from_file_location("enrich_manifest", ENRICH_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class _RowsDataset:
    def __init__(self, writers: list[str]) -> None:
        self.rows = [{"writer_id": writer, "record_id": f"{i}/{i}.png"} for i, writer in enumerate(writers)]

    def __len__(self) -> int:
        return len(self.rows)


def test_enrich_manifest_fills_reference_paths(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    output = tmp_path / "out.jsonl"
    rows = [
        {
            "record_id": "u1/a.png",
            "char_id": "一",
            "writer_id": "w1",
            "style_family_id": "w1__kai",
            "style_unit_id": "w1__work1__kai",
            "script_label": "楷書",
        },
        {
            "record_id": "u1/b.png",
            "char_id": "二",
            "writer_id": "w1",
            "style_family_id": "w1__kai",
            "style_unit_id": "w1__work1__kai",
            "script_label": "楷書",
        },
        {
            "record_id": "u2/c.png",
            "char_id": "三",
            "writer_id": "w1",
            "style_family_id": "w1__xing",
            "style_unit_id": "w1__work2__xing",
            "script_label": "行書",
        },
    ]
    _write_jsonl(source, rows)

    subprocess.run(
        [
            sys.executable,
            str(ENRICH_SCRIPT),
            "--input",
            str(source),
            "--output",
            str(output),
            "--image-root",
            "/images",
            "--content-cache-root",
            "/cache",
            "--refs-per-row",
            "2",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    enriched = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    first = enriched[0]
    assert first["image_path"] == "/images/u1/a.png"
    assert len(first["ref_image_paths"]) == 2
    assert "/images/u1/a.png" not in first["ref_image_paths"]
    assert all(not path.endswith("/a.png") for path in first["ref_image_paths"])
    assert first["reference_ids"] == [
        path.removeprefix("/images/") for path in first["ref_image_paths"]
    ]
    ref_chars = {
        row["char_id"]
        for row in rows
        if f"/images/{row['record_id']}" in first["ref_image_paths"]
    }
    assert "一" not in ref_chars


def test_choose_reference_rows_excludes_query_char() -> None:
    module = _load_enrich_module()
    rows = [
        {"record_id": "a/1.png", "char": "", "target_char": "一", "writer_id": "w", "style_unit_id": "u"},
        {"record_id": "a/2.png", "char_id": "一", "writer_id": "w", "style_unit_id": "u"},
        {"record_id": "a/3.png", "char_id": "二", "writer_id": "w", "style_unit_id": "u"},
    ]
    by_unit, by_writer = module._build_reference_indexes(rows)
    refs = module.choose_reference_rows(
        rows[0],
        by_unit=by_unit,
        by_writer=by_writer,
        refs_per_row=4,
        ref_scope="same_unit_or_writer",
        seed=0,
    )
    assert [ref["record_id"] for ref in refs] == ["a/3.png"]


def test_writer_balanced_sampler_uses_inverse_writer_frequency() -> None:
    dataset = _RowsDataset(["a", "a", "a", "b"])
    sampler = build_manifest_train_sampler(
        dataset,
        data_cfg={"sampling": {"mode": "writer_balanced"}},
        train_cfg={},
        seed=123,
    )
    assert isinstance(sampler, WeightedRandomSampler)
    weights = sampler.weights.to(dtype=torch.float32).tolist()
    assert weights[:3] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert weights[3] == pytest.approx(1.0)


def test_writer_cap_sampler_limits_per_writer() -> None:
    dataset = _RowsDataset(["a", "a", "a", "a", "b", "b"])
    sampler = build_manifest_train_sampler(
        dataset,
        data_cfg={"sampling": {"mode": "writer_cap", "max_samples_per_writer_per_epoch": 2}},
        train_cfg={},
        seed=123,
    )
    assert sampler is not None
    indices = list(sampler)
    writers = [dataset.rows[i]["writer_id"] for i in indices]
    assert len(indices) == 4
    assert Counter(writers) == {"a": 2, "b": 2}
