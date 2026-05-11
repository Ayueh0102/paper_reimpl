# Stage A v1 — Real TTF Cross-Font Pretrain

**Goal:** first real Stage A run on the lab server, replacing the synthetic
shakedown. Trains FontDiffuser on cross-font pairs rendered on-the-fly from
13 open-source Chinese fonts. Produces the warm-start ckpt for Stage B.

## Run

| field | value |
|---|---|
| Date | 2026-05-11 |
| Host | WIN-C20DRJGJ4S4 (lab server) |
| GPU | NVIDIA RTX 6000 Ada Generation 48 GB (cuda:0) |
| Driver | CUDA 12.8, 573.42 |
| torch | 2.11.0+cu128 |
| Config | `configs/{train_stage_a_ttf_v1, data_stage_a_ttf, model}.yaml` |
| Data | TTFCrossFontPairDataset over 13 fonts in `data_snapshot/fonts_free/` |
| Char vocabulary | **4 865** chars (intersection of CJK Basic U+4E00..U+9FFF across all 13 fonts) |
| Steps | 5 000 |
| Batch size | 16 |
| LR | 1e-4 (AdamW) |
| Grad clip | 1.0 |
| CFG drop | 0.1 (white-out content + style, per official) |
| SCR weight | 0.0 (Stage A skips contrastive head) |
| Perceptual weight | 0.01 (VGG-16) |
| Offset L1 weight | 0.5 (RSI) |
| Diffusion | T=1000, β∈[1e-4, 2e-2] linear, eps prediction |
| Wall clock | 29 min (≈6.5 min char discovery + ≈22.5 min training) |
| Training throughput | ≈3.5 step / s (≈55 samples / s on 128² + VGG) |
| ckpt | `outputs/stage_a_ttf_v1/fontdiffuser_last.pt` (117 MB) |
| ckpt sha256 | `C56D0022A65823026E4977DDABAC63D136BA8933C4FC6EB14CE05D2B20D88545` |
| Log | `reports/stage_a_ttf_v1_log.txt` (56 lines) |
| Char cache | `reports/ttf_supported_chars_128px_0.85.json` (4 865 entries) |

## Loss trajectory (every 100 steps)

| step | total | simple | perc | offset |
|---:|---:|---:|---:|---:|
| 0    | 2.1235 | 1.1814 | 94.1339 | 0.0015 |
| 100  | 0.5491 | 0.0473 | 50.1813 | 0.0001 |
| 200  | 0.3144 | 0.0534 | 26.0959 | 0.0001 |
| 500  | 0.2365 | 0.0347 | 20.1800 | 0.0001 |
| 1000 | 0.2810 | 0.0167 | 26.4321 | 0.0000 |
| 1500 | 0.1453 | 0.0301 | 11.5157 | 0.0000 |
| 2000 | 0.1756 | 0.0184 | 15.7161 | 0.0000 |
| 2500 | 0.2885 | 0.0152 | 27.3219 | 0.0001 |
| 3000 | 0.1329 | 0.0223 | 11.0507 | 0.0000 |
| 3500 | 0.2006 | 0.0211 | 17.9539 | 0.0000 |
| 4000 | 0.1874 | 0.0125 | 17.4914 | 0.0001 |
| 4500 | 0.2223 | 0.0131 | 20.9222 | 0.0000 |
| 4700 | 0.1420 | 0.0148 | 12.7199 | 0.0000 |
| 4900 | 0.1534 | 0.0177 | 13.5725 | 0.0000 |

- Total loss drops ≈14× from step 0 → step 100, then descends with high
  variance to a 0.14–0.22 band by step 1500.
- ``loss_simple`` (epsilon MSE) collapses to <0.02 by step 200 and stays
  there — the model has learned the denoising target quickly.
- The remaining variance comes from VGG perceptual (uses cross-font
  pairs where target style differs from source content, so per-batch
  difficulty varies).
- No NaN / Inf across all 50 logged windows.

## Bugs fixed during this run

- `dataset.py` `supported_chars_cache` relative-path resolution was
  unspecified; cwd was `papers/01_fontdiffuser/` so the cache landed at
  the nested `papers/01_fontdiffuser/papers/01_fontdiffuser/outputs/…`.
  Default location now resolves under `<fonts_root>/` so the cache
  travels with the data across backends. (commit, see below)
- `render_glyph` re-parsed the TTF file on every call. Added an
  `functools.lru_cache` on `(ttf_path, point_size)`, ~5× speedup measured
  on Mac. (commit `3a5aaee`)

## What this validates (vs. shakedown)

- [x] TTFCrossFontPairDataset works end-to-end on Windows + CUDA.
- [x] Char discovery (rendering 273 k probe glyphs through 13 fonts)
      completes in ~6.5 min on the server, caches to JSON, subsequent
      starts hit the cache in <1 s.
- [x] Tofu/.notdef rejection via bitmap-hash comparison produced a
      sensible 4 865-char intersection across kai/xing/cao/hei/ming/decor.
- [x] FontDiffuser fwd+bwd+ckpt save on real cross-font triples.
- [x] CFG dropout + VGG perceptual + RSI offset all contribute, no NaN.

## Open items before Stage B / longer Stage A

1. **Sample qualitative review** — pull a `sample_grid.png` rendered
   with `sample_ddim` from the v1 ckpt; eye-check whether the model is
   producing legible characters (not just denoising).
2. **Longer Stage A run** — 5000 step is a milestone, not a converged
   pretrain. Paper-faithful would be 50 k+ steps. Decide on a v2
   schedule before committing GPU days.
3. **Stage B wiring** — switch `data_*.yaml` to manifest mode, pull
   `content_fields_cache` from mother repo to lab server (583 MB scp),
   warm-start from `stage_a_ttf_v1/fontdiffuser_last.pt`.
4. **DataLoader workers > 0 on Windows** — currently num_workers=0.
   With workers + render cache, throughput could likely 2-3×.
