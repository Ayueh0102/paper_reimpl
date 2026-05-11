# Blind reimpl decision log — 03 IF-Font (NeurIPS 2024)

> **PHASE 2 STATUS**: All 7 P0 deviations from `reports/github_diff.md`
> have been resolved by the Phase-2 surgery. See the "Phase 2 corrections"
> section at the bottom of this file for the per-item diff. The "Decision
> Log" below documents the original (Phase-1, blind) state for historical
> reference; current decisions live in `paper_notes/03.md`.

This is a **blind reimplementation** done from the project's paper notes
without inspecting the official GitHub
(`github.com/Stareven233/IF-Font`). Every non-trivial design choice is
flagged as either `[paper-cited]` (verbatim from the note) or
`[guessed-*]` (filled in to make the implementation runnable).

## Source material used

- `/Users/Ayueh/Documents/Obsidian Vault/research/papers/033_IF-Font表意描述序列字體_NeurIPS2024.md`
- `/Users/Ayueh/Char/paper_reimpl/reports/phase0_spec_table.md` row 03
- IDS dictionary from `~/Char/datasets/ids/derived/cns_unicode_ids.tsv`
  with helper `~/Char/datasets/ids/scripts/lookup_ids.py`
  (paper says "IDS" but doesn't pin a dictionary; this is the dataset we
  already curated and it covers ≥99.99 % of GB2312/Big5 chars).

## Decision Log

### Architecture

1. **[paper-cited §"訓練配置"] VQGAN codebook size = 256, downsample = 8.**
   Implemented in `model.py:VQTokenizerConfig` (`codebook_size=256`,
   `channel_mult=(1, 2, 2, 4)` → 3 strided downsamples → factor 8).

2. **[paper-cited §"訓練配置"] AR Transformer = 10 blocks, 8 heads,
   dim 384, 2 self-attn + 1 cross-attn per block.** Implemented in
   `model.py:_DecoderBlock` and `IFFontConfig` defaults (`n_blocks=10`,
   `n_heads=8`, `d_model=384`, `n_self_attn_per_block=2`).

3. **[paper-cited] IDS replaces source glyph as the content signal.**
   Implemented in `ids.py:IDSTokenizer` and the `IDSTextEncoder` →
   cross-attention path in `model.py:IFFont.build_context`.

4. **[guessed-because-paper-vague] VQ encoder/decoder layer widths.**
   Paper only mentions "VQGAN" + downsample factor. We chose
   `base_channels=64`, `channel_mult=(1,2,2,4)`. Rationale: 4 stages with
   3 downsamples gives the required factor 8; 64→256 channel ladder is
   the smallest commonly-used VQGAN configuration that gives the codebook
   enough representational room.

5. **[guessed-because-paper-vague] VQ codebook update rule = EMA +
   commitment loss + straight-through.** Paper says "VQGAN" without
   specifying the original VQ-VAE update rule vs the EMA variant. We use
   EMA (decay=0.99) because it is more stable on small batches, which is
   important for Stage A on Ernantang's 4659 chars (paper trained on a
   larger custom dataset).

6. **[guessed-because-paper-vague] IDS text encoder = 2-layer Transformer
   encoder, 4 heads, 384 dim.** Paper says "text encoder" but pins no
   architecture. We chose a lightweight Transformer encoder; depth 2
   is the minimum that allows IDC ↔ leaf token interactions while keeping
   parameter count negligible compared to the 10-block AR decoder.

7. **[guessed-because-paper-vague] Reference tokens are flattened into a
   single sequence `[B, N*256, d_model]` and concatenated *after* the
   IDS context.** Paper does not specify ref-token packing or ordering.
   We use a simple concatenation and let the decoder's cross-attention
   learn to weight IDS vs ref tokens. An attention-mask separator
   approach is a possible future refinement.

8. **[guessed-from-Ho2022] CFG dropout probability = 0.1.** Paper does
   not state CFG hyperparameters; 0.1 is the well-known Ho & Salimans
   value used by every conditional diffusion / AR re-implementation we
   are aware of.

9. **[guessed] BOS token convention — codebook index `K = codebook_size`.**
   Paper does not name a start token. We add an extra row at the top of
   the AR token embedding table (`nn.Embedding(K+1, d_model)`) and use
   index `K` as BOS. This avoids reserving a codebook slot.

10. **[guessed] AR scan order = raster (row-major).** Paper does not
    specify; raster is the default for VQGAN-AR transformers (e.g.
    Esser 2021 / DALL·E 1). Diagonal / Z-order scan are obvious future
    ablations.

### Losses

11. **[paper-cited] Main loss = cross-entropy on next-VQ-token
    prediction.** Implemented in `train.py:compute_loss` (mean reduction
    over `[B, N]` positions). The reduction choice matches Ho-style "mean
    over batch *and* spatial" which is the right scale for AdamW lr at
    1e-4 with the paper's batch_size=128.

12. **[paper-cited] Stage A loss = VQ commitment + reconstruction MSE.**
    Implemented in `train.py:compute_loss` via the `(ce_weight,
    vq_weight, recon_weight)` triple. Stage A YAML sets
    `ce_weight=0, vq_weight=1, recon_weight=1`; Stage B/C sets
    `ce_weight=1` and the others to 0.

13. **[guessed-because-paper-vague] Commitment coefficient β = 0.25.**
    Paper does not pin β; we use the original VQ-VAE default.

### Training / data

14. **[paper-cited §"訓練配置"] AdamW + warmup + cosine schedule, batch
    128, 15 epochs.** Paper-cited numbers; our train YAMLs default to
    smaller batch (16/32) so a single mid-range GPU can fit the model
    plus the ref VQ encode pass.

15. **[guessed-because-paper-vague] AdamW betas = (0.9, 0.95).** Paper
    does not pin Adam betas; 0.9 / 0.95 is the standard
    Transformer-decoder choice (e.g. GPT-2 / VQGAN papers).

16. **[guessed-because-paper-vague] Weight decay = 0.05.** Same logic
    — Transformer-decoder default.

17. **[guessed] Gradient clip norm = 1.0.** Standard for AR
    Transformers; paper does not specify.

18. **[note-from-CLAUDE.md] IDS dictionary source = CHISE-derived
    `cns_unicode_ids.tsv`.** Paper cites "linguistically-defined IDS" but
    doesn't pin a dictionary release. We use the dataset already in
    `~/Char/datasets/ids/derived/`, accessed via `lookup_ids.get_ids`.
    The dataset README claims 99.99 % coverage of GB2312 + Big5 chars,
    which is more than enough for Ernantang's 4659-char target.

19. **[guessed-because-paper-vague] Synthetic IDS for smoke/dry-run.**
    The smoke test (and the synthetic dataset path) cannot rely on the
    external IDS TSV being present. We synthesise a deterministic IDS
    per row using `synthetic_ids_for_index(index)` — one of the 12 IDC
    chars plus two pseudo-leaf CJK chars in U+4E00..U+51E8. This keeps
    the IDS → text encoder gradient path active.

20. **[guessed] IDS max length = 32 tokens.** Long IDS like 「龘」
    have ~10–14 chars; 32 leaves headroom plus BOS/EOS. Paper does
    not pin this.

### Engineering / shared-runtime compatibility

21. **[shared-runtime-contract] Synthetic batch uses
    `paper_reimpl_shared.runner.smoke.make_synthetic_batch`.** The smoke
    test does *not* pass `char_id`/`writer_id` into the model — IF-Font
    only consumes (image, refs, IDS). This is intentional and matches
    paper §3 which says the system is "source-glyph-free".

22. **[shared-runtime-contract] `train.main(args, *, data_cfg, model_cfg,
    train_cfg, paths)` matches the entrypoint signature.** Validated
    against `papers/01_fontdiffuser/src/fontdiffuser/train.py`.

## Verification checklist (done criteria)

- [x] `uv sync` produces a working venv.
- [x] `uv run pytest tests/test_smoke.py -x` GREEN (6 tests).
- [x] `uv run python -m paper_reimpl_shared.runner.entrypoint --paper if_font --dry-run --synthetic --device cpu --train ... --model ... --data ... --data-backend mac_symlink` finishes 1 step.
- [x] `paper_notes/03.md` ≥ 800 words (≈ 970 words).
- [x] `reports/blind_impl.md` contains ≥ 5 `[guessed-*]` entries
      (currently 13 `[guessed-*]` items).

## Known gaps / nice-to-haves for Phase 2

- KV caching in `TransformerARDecoder.sample` — current implementation
  recomputes the full prefix every step. Fine for smoke; fix before
  long sequence eval.
- Real Stage A VQGAN adversarial loss (paper note says "VQGAN"; we use
  only the reconstruction + commitment terms, no discriminator).
- Tied weights between AR decoder token embedding and codebook
  embeddings — could improve token quality, but paper doesn't say to do
  it.
- Top-k / top-p AR sampling at inference (current is temperature
  multinomial).
- IDS data augmentation (e.g. randomly swap equivalent IDS forms) for
  robustness — useful nice-to-have, paper does not mention.

---

## Phase 2 corrections (2026-05-11)

Phase 2 audits Phase-1 against the official Stareven233/IF-Font repository
(`reports/github_diff.md`). The 7 P0 deviations are listed below with the
files touched. **All P0 items are now resolved.** P1/P2 items partially
addressed; the remaining ones are tracked in `paper_notes/03.md §7`.

### P0 #1 — Decoder block: 1 self-attn (not 2) — ✅ FIXED

- Removed `IFFontConfig.n_self_attn_per_block` (was default 2).
- `src/if_font/model.py:_DecoderBlock`: now `attn + attn_cross + mlp`
  (one of each), matches official `Block2.forward` in
  `iffont/modules/nanogpt.py:159-163`.
- Added test `test_decoder_block_has_one_self_attn` to lock the invariant.

### P0 #2 — Frozen pretrained VQGAN (not from-scratch) — ✅ FIXED

- `src/if_font/model.py:VQTokenizerAdapter` replaces the trained-from-scratch
  `VQTokenizer`. Defaults: `embedding_dim=4`, `codebook_size=256`,
  `downsample_factor=8`, `in_channels=3` (RGB).
- The adapter is frozen at construction via `requires_grad_(False)`,
  `.eval()`, and a `train` override (matches official
  `iffont/data/adapter.py:54`).
- `VQTokenizerAdapter.from_pretrained_compvis(path)` loads the real
  CompVis weights via `taming-transformers` + `omegaconf` (added to
  `pyproject.toml` as the `pretrained-vqgan` extra).
- Smoke / CI use a random-weight stub adapter with the correct shapes;
  this lets the rest of the model train end-to-end without the heavy
  pretrained-weights download.
- Stage A (VQGAN pretraining) is dropped — see Phase-2 train YAMLs.
- The deprecated `VectorQuantizer` is kept as an import-error stub.

### P0 #3 — StyleEncoder + 3SA — ✅ FIXED

- `src/if_font/model.py:StyleEncoder` implements CNN stem + coverage-pool
  + 3SA cross-attention. Mirrors official
  `iffont/modules/encoder.StyleEncoder.forward` (lines 601-622).
- The CNN stem (`_ConvBlock` + `_ResBlock`, instance-norm + reflect-pad)
  matches the layer ladder in `encoder._init_enc`.
- `_structure_style_aggregation` runs IDS-query → ref-feature K/V
  cross-attention with per-head QK-LN (`encoder.py:587-599`).

### P0 #4 — MoCo wrapper + sup_cl contrastive loss — ✅ FIXED

- `src/if_font/model.py:MoCoWrapper` carries two StyleEncoders
  (`enc` query + `enc_m` momentum), 2-layer MLP projector + predictor,
  and cosine-scheduled momentum update.
- `src/if_font/losses.py:sup_cl` is a port of official
  `iffont/modules/losses.sup_cl` (Khosla 2020 SupCon).
- `src/if_font/train.py:MoCoCache` is a bounded FIFO of (cl, font_id)
  pairs; matches `models/net2net_model.CacheManagerCL` (size=10).
- `compute_loss` returns `sq + sup_cl_weight * sup_cl` where
  `sup_cl_weight=0.5` captures the official `/ 2`.

### P0 #5 — Coverage-similarity ref routing — ✅ FIXED

- `IFFont.compute_coverage` (class method) and the same algorithm inlined
  in `dataset.py:_compute_coverage_row` implement the IDC-anchored
  longest-common-substring score from official
  `IDSEncoder.coverage` (encoder.py:307-357).
- The collate computes `coverage_sim [B, N]` at batch construction time
  and the model passes it through StyleEncoder for the `x_g` softmax weights.

### P0 #6 — BabelStone + ids_iffont (not CHISE) — ✅ FIXED

- `src/if_font/ids.py:IDSResolver` reads
  `~/Char/datasets/ids/cn_mainland/babelstone_cjk_ids.txt`
  (97058 entries, vendored from third_party) and
  `~/Char/datasets/ids/cn_mainland/ids_iffont.txt`
  (165 entries, vendored from third_party).
- Supports `level='radical'` (default, matches official `base.yaml:70`)
  and `level='stroke'` (used by coverage similarity).
- `IDSTokenizer.fit_from_resolver` builds the leaf vocab off the
  resolver's char set; this is the production path. The Phase-1 CHISE
  fallback via `lookup_ids.py` is kept only as a legacy `ids_lookup_path`
  config knob.

### P0 #7 — RGB (not grayscale) — ✅ FIXED

- `IFFontConfig.in_channels = 3` and `VQTokenizerConfig.in_channels = 3`
  by default.
- `data_stage_*.yaml` switched to RGB (Phase-1 yamls were grayscale).
- `dataset.py:_to_rgb` replicates 1-channel synthetic images to 3
  channels at the collate boundary; the smoke test uses the existing
  shared `make_synthetic_batch` with `in_channels=3`.

### Partial P1 corrections

- **A6 (prefix-prepended IDS)**: implemented. `TransformerARDecoder.forward`
  now does `tok_emb = cat([ids_embed, wte(idx)])` then slices
  `logits[:, ids_len-1:]`, mirroring official `nanogpt.GPT.forward`.
- **A7 (QK-LayerNorm)**: implemented in `_CausalSelfAttention` and
  `_CrossAttention`. Both use `F.scaled_dot_product_attention` (Flash).
  (DropKey mask is not yet implemented — tracked as a remaining P1.)
- **A8 (weight tying wte ↔ lm_head)**: implemented via
  `self.wte.weight = self.lm_head.weight` in `TransformerARDecoder.__init__`.
- **S1 (OneCycleLR + 2-group AdamW)**: implemented in
  `train._configure_optimizers` and `train.main`.
- **L1 (drop VQ/recon losses)**: implemented. Phase-1's
  `(ce_weight, vq_weight, recon_weight)` triple is gone; only
  `sup_cl_weight` remains as a knob.
- **L2 (no CFG)**: removed. `cfg_drop_prob` and the all-masked-row guard
  are dropped — the official model never uses CFG.

### Still outstanding (deferred)

These are non-blockers for getting the trainer wired correctly but are
known divergences:

- **DropKey attention mask** (A7 partial): not implemented; the official
  `modules.blocks.DropKeyMask` Bernoulli-drops keys at training time
  before SDPA. Phase 2 uses vanilla SDPA. P1 #9 in github_diff.
- **Triplet-equivalent IDS augmentation** (D3): not implemented. Official
  randomly swaps equivalent IDS forms during `IDSEncoder.embed`. P1 #15.
- **Pre-tokenised HDF5 pipeline** (C1): not implemented. Official
  precomputes target+ref VQ indices into HDF5 and the training loop
  never touches raw pixels. Phase 2 still encodes refs on the fly inside
  `IFFont.forward`. P1 #17.
- **DDP all-gather for sup_cl** (A4 detail): single-GPU only. Official
  `Net2NetModel.training_step` all-gathers cl features across DDP ranks.
- **Mixed precision** (S3): not yet wired into our shared runner; the
  official runs `precision: 16-mixed`.

### Files modified in Phase 2

```
src/if_font/__init__.py                  (re-exports)
src/if_font/configs/data_stage_a.yaml    (in_channels gone — RGB at collate)
src/if_font/configs/data_stage_b.yaml    (max_refs 3, in_channels at model level)
src/if_font/configs/data_stage_c.yaml    (same as Stage B)
src/if_font/configs/model.yaml           (Phase-2 hyperparams)
src/if_font/configs/train_stage_a_ttf.yaml (no-op shim; Stage A dropped)
src/if_font/configs/train_stage_b_midtrain.yaml (sup_cl, OneCycle, no CE/VQ/recon)
src/if_font/configs/train_stage_c_ernantang.yaml (same recipe)
src/if_font/dataset.py                   (RGB collate, coverage_sim, font_id)
src/if_font/ids.py                       (IDSResolver: BabelStone + ids_iffont)
src/if_font/losses.py                    (NEW: sq + sup_cl)
src/if_font/model.py                     (VQTokenizerAdapter, StyleEncoder, MoCo, 1+1+1 block, prefix AR)
src/if_font/sample.py                    (coverage_sim arg, RGB output)
src/if_font/train.py                     (2-group AdamW + OneCycleLR + MoCoCache)
tests/test_smoke.py                      (Phase-2 invariants)
pyproject.toml                           (pretrained-vqgan optional extra)
paper_notes/03.md                        ([REVISED PER PHASE 2])
~/Char/datasets/ids/cn_mainland/babelstone_cjk_ids.txt (NEW: vendored from third_party)
~/Char/datasets/ids/cn_mainland/ids_iffont.txt         (NEW: vendored from third_party)
```

### Verification

```
$ uv run ruff check src/ tests/
All checks passed!

$ uv run pytest tests/test_smoke.py -x
14 passed in 1.11s

$ uv run python -m paper_reimpl_shared.runner.entrypoint \
    --paper if_font --dry-run --synthetic --device cpu \
    --train src/if_font/configs/train_stage_b_midtrain.yaml \
    --model src/if_font/configs/model.yaml \
    --data src/if_font/configs/data_stage_b.yaml \
    --data-backend mac_symlink
[if_font] device=cpu max_steps=1 bs=32 lr=1.44e-04 sup_cl_weight=0.5
          codebook=256 d_model=384 blocks=10 n_refs=3 vqgan_pretrained=False
[if_font] epoch=0 step=0 total=8.5214 sq=5.5454 cl=5.9521
[if_font] done; final_step=1 dry_run=True
```
