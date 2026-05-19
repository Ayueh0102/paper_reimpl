"""Enrich a split JSONL with absolute image_path + content_npz + vocab IDs.

The split manifests carry only relative ``record_id`` (e.g. "1000/1000_78.png")
and ``char_id`` (e.g. "朗"). The training pipeline's legacy loader expects
``image_path`` and ``content_npz`` keys plus integer label IDs. This script
fills both in for the lab-server backend.

Usage:
    python scripts/enrich_manifest.py \\
        --input /path/to/split.jsonl \\
        --output /path/to/split_enriched.jsonl \\
        --image-root "$PR_IMAGE_ROOT" \\
        --content-cache-root "$PR_DATA_ROOT/content_fields_cache" \\
        --image-size 256
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _first_nonempty(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _row_char(row: dict[str, Any]) -> str:
    return _first_nonempty(row, ("char", "target_char", "char_id", "label"))


def _row_unit(row: dict[str, Any]) -> str:
    return _first_nonempty(row, ("style_unit_id", "style_family_id"))


def _row_writer(row: dict[str, Any]) -> str:
    return _first_nonempty(row, ("writer_id", "writer"))


def _row_record_id(row: dict[str, Any]) -> str:
    return str(row["record_id"]).replace("\\", "/")


def _build_reference_indexes(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_writer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        unit = _row_unit(row)
        writer = _row_writer(row)
        if unit:
            by_unit[unit].append(row)
        if writer:
            by_writer[writer].append(row)
    for index in (by_unit, by_writer):
        for key, values in index.items():
            index[key] = sorted(values, key=_row_record_id)
    return by_unit, by_writer


def _candidate_refs(
    row: dict[str, Any],
    *,
    by_unit: dict[str, list[dict[str, Any]]],
    by_writer: dict[str, list[dict[str, Any]]],
    ref_scope: str,
) -> list[dict[str, Any]]:
    pools: list[list[dict[str, Any]]] = []
    if ref_scope in {"same_unit", "same_unit_or_writer"}:
        pools.append(by_unit.get(_row_unit(row), []))
    if ref_scope in {"same_writer", "same_unit_or_writer"}:
        pools.append(by_writer.get(_row_writer(row), []))

    query_record = _row_record_id(row)
    query_char = _row_char(row)
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for pool in pools:
        for candidate in pool:
            record_id = _row_record_id(candidate)
            if record_id == query_record or record_id in seen:
                continue
            if _row_char(candidate) == query_char:
                continue
            seen.add(record_id)
            candidates.append(candidate)
    return candidates


def choose_reference_rows(
    row: dict[str, Any],
    *,
    by_unit: dict[str, list[dict[str, Any]]],
    by_writer: dict[str, list[dict[str, Any]]],
    refs_per_row: int,
    ref_scope: str,
    seed: int,
) -> list[dict[str, Any]]:
    if refs_per_row <= 0:
        return []
    candidates = _candidate_refs(
        row,
        by_unit=by_unit,
        by_writer=by_writer,
        ref_scope=ref_scope,
    )
    if len(candidates) <= refs_per_row:
        return candidates
    rng = random.Random(f"{seed}:{_row_record_id(row)}")
    return rng.sample(candidates, refs_per_row)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--image-root", required=True,
                   help="Base dir for calligraphy renders, e.g. D:/Char/char_full/public/images/WMF")
    p.add_argument("--content-cache-root", required=True,
                   help="Content cache root, e.g. D:/Char/.../content_fields_cache")
    p.add_argument("--image-size", type=int, default=256,
                   help="Image size for content npz subdir (128 or 256).")
    p.add_argument("--refs-per-row", type=int, default=16,
                   help="Number of reference images to attach per row.")
    p.add_argument(
        "--ref-scope",
        choices=["same_unit", "same_writer", "same_unit_or_writer"],
        default="same_unit_or_writer",
        help="Reference pool. same_unit_or_writer tries the unit first, then same writer.",
    )
    p.add_argument("--ref-seed", type=int, default=42,
                   help="Seed for deterministic per-row reference selection.")
    p.add_argument(
        "--overwrite-refs",
        action="store_true",
        help="Replace non-empty ref_image_paths/reference_ids instead of filling only empty rows.",
    )
    args = p.parse_args()

    image_root = str(args.image_root).rstrip("/\\")
    content_root = str(args.content_cache_root).rstrip("/\\")
    size_dir = str(args.image_size)

    rows = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[enrich] loaded {len(rows)} rows from {args.input}")

    # Build vocab maps.
    writer_to_id = {}
    style_family_to_id = {}
    style_unit_to_id = {}
    char_to_id = {}
    SCRIPT_ORDER = ["楷書", "行書", "草書", "小篆"]
    script_to_id = {s: i for i, s in enumerate(SCRIPT_ORDER)}
    for row in rows:
        for k, m in [
            (str(row.get("writer_id", "")), writer_to_id),
            (str(row.get("style_family_id", "")), style_family_to_id),
            (str(row.get("style_unit_id", "")), style_unit_to_id),
            (_row_char(row), char_to_id),
        ]:
            if k and k not in m:
                m[k] = len(m)
        s = str(row.get("script_label", ""))
        if s and s not in script_to_id:
            script_to_id[s] = len(script_to_id)

    print(f"[enrich] vocab: writers={len(writer_to_id)} "
          f"style_family={len(style_family_to_id)} "
          f"style_unit={len(style_unit_to_id)} chars={len(char_to_id)} "
          f"scripts={len(script_to_id)}")
    by_unit, by_writer = _build_reference_indexes(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filled_refs = 0
    empty_refs = 0
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            record_id = _row_record_id(row)
            char = _row_char(row)
            hex_code = f"0x{ord(char):04x}" if char else "0x0000"
            enriched = dict(row)
            enriched["image_path"] = f"{image_root}/{record_id}"
            enriched["content_npz"] = f"{content_root}/{size_dir}/{hex_code}.npz"
            enriched["writer_label_id"] = writer_to_id.get(str(row.get("writer_id", "")), 0)
            enriched["style_family_label_id"] = style_family_to_id.get(
                str(row.get("style_family_id", "")), 0
            )
            enriched["unit_label_id"] = style_unit_to_id.get(
                str(row.get("style_unit_id", "")), 0
            )
            enriched["char_label_id"] = char_to_id.get(char, 0)
            enriched["script_label_id"] = script_to_id.get(
                str(row.get("script_label", "")), 0
            )
            enriched["writer_vocab_size"] = len(writer_to_id)
            enriched["style_family_vocab_size"] = len(style_family_to_id)
            enriched["unit_vocab_size"] = len(style_unit_to_id)
            enriched["char_vocab_size"] = len(char_to_id)
            enriched["script_vocab_size"] = len(script_to_id)
            if args.overwrite_refs or not enriched.get("ref_image_paths"):
                refs = choose_reference_rows(
                    row,
                    by_unit=by_unit,
                    by_writer=by_writer,
                    refs_per_row=args.refs_per_row,
                    ref_scope=args.ref_scope,
                    seed=args.ref_seed,
                )
                reference_ids = [_row_record_id(ref) for ref in refs]
                enriched["reference_ids"] = reference_ids
                enriched["ref_image_paths"] = [
                    f"{image_root}/{reference_id}" for reference_id in reference_ids
                ]
                enriched["reference_policy"] = {
                    "ref_mode": args.ref_scope,
                    "reference_pool": "input_manifest",
                    "requested_ref_mode": args.ref_scope,
                    "refs_per_row": args.refs_per_row,
                    "exclude_query_char": True,
                }
            if enriched.get("ref_image_paths"):
                filled_refs += 1
            else:
                empty_refs += 1
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    print(f"[enrich] wrote {len(rows)} enriched rows to {args.output}")
    print(f"[enrich] ref_image_paths non_empty={filled_refs} empty={empty_refs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
