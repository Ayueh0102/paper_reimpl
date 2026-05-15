"""Enrich a split JSONL with absolute image_path + content_npz + vocab IDs.

The split manifests carry only relative ``record_id`` (e.g. "1000/1000_78.png")
and ``char_id`` (e.g. "朗"). The training pipeline's legacy loader expects
``image_path`` and ``content_npz`` keys plus integer label IDs. This script
fills both in for the lab-server backend.

Usage:
    python scripts/enrich_manifest.py \\
        --input /path/to/split.jsonl \\
        --output /path/to/split_enriched.jsonl \\
        --image-root "D:/Char/char_full/public/images/WMF" \\
        --content-cache-root "D:/Char/ayueh/paper_reimpl/data_snapshot/content_fields_cache" \\
        --image-size 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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
            (str(row.get("char_id", "")), char_to_id),
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            record_id = str(row["record_id"]).replace("\\", "/")
            char = str(row.get("char_id", ""))
            hex_code = f"0x{ord(char):04X}" if char else "0x0000"
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
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    print(f"[enrich] wrote {len(rows)} enriched rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
