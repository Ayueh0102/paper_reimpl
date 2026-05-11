# 02_hfh_font — Stage A Shakedown (cuda:1, parallel)

**Goal:** validate the Phase-3 pipeline on cuda:1 while 01_fontdiffuser
Stage A v3 owns cuda:0. Synthetic-only — TTF cross-font adapter for 02
is deferred until v3 completes.

## Run

| field | value |
|---|---|
| Date | 2026-05-11 |
| GPU | RTX 6000 Ada (cuda:1) |
| Steps | 200 |
| Batch size | 8 |
| Data | synthetic, `synthetic_length=3200` (so single-epoch loop reaches 200 steps) |
| Model | 25.38 M params (FontDiffuser is larger) |
| Diffusion target | x0 (HFH-Font convention; not eps like 01) |
| Wall time | ~3 min |

## Loss trajectory

| step | loss |
|---:|---:|
| 0   | 0.2354 |
| 20  | 0.0585 |
| 40  | 0.0518 |
| 60  | 0.0511 |
| 80  | 0.0493 |
| 100 | 0.0456 |
| 120 | 0.0480 |
| 140 | 0.0422 |
| 160 | 0.0412 |
| 180 | 0.0406 |
| 199 | 0.0392 |

6× drop over 200 steps. No NaN. Note that loss is x0 MSE (not eps like 01),
so absolute values are not directly comparable.

## What this validates

- [x] 02 venv with torch 2.11.0+cu128 on cuda:1
- [x] pytest smoke (3/3 PASSED)
- [x] HFH-Font model build + fwd/bwd + AdamW
- [x] Synthetic dataset path (multi-ref `n_refs=4`, multi-channel content
      `[bitmap, sdf, skeleton]`)
- [x] Per-step loss logging via Python `logging`

## Phase 3 gaps in 02 (must fix before any long run)

1. **No ckpt save** at end of training loop. `outputs/` directory was
   never created. Need to add a `torch.save(...)` after the loop and
   handle `--dry-run` correctly.
2. **No max_epochs loop** — `for batch in loader` is single-pass, so
   `max_steps` is silently capped by `synthetic_length / batch_size`.
   Shakedown worked around by setting `synthetic_length=3200`, but for a
   long synthetic / manifest run we need an outer epoch loop.
3. **No TTF cross-font dataset adapter** — 02's `build_dataset` only
   knows `synthetic` and `manifest` modes. To use the shared
   `TTFCrossFontPairDataset` we need a `source: ttf` branch in
   `dataset.py` like 01 has.

## Defer until 01 v3 finishes

01 owns cuda:0 for ~6 hours; spend that time fixing the three gaps above
on Mac (no GPU needed for the fixes themselves), then re-run shakedown
to verify.
