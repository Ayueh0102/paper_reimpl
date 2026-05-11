# Blind Implementation — VQ-Font (04)

Decision log for the Phase 1 blind reimplementation. Every non-trivial
choice is tagged either `[paper-cited <source>]` when sourced from the
Obsidian paper note or the Phase 0 spec table, `[guessed-because-paper-vague]`
when we filled in a gap, or `[blind-impl-convention]` for paper-neutral
tooling choices.

## Source material consulted

1. `/Users/Ayueh/Documents/Obsidian Vault/research/papers/036_VQ-Font結構感知字體_AAAI2023.md`
2. `~/Char/paper_reimpl/reports/phase0_spec_table.md`, row 04
3. `~/Char/paper_reimpl/AGENTS.md`
4. `~/Char/paper_reimpl/shared/src/paper_reimpl_shared/...`
5. `~/Char/paper_reimpl/docs/REVIEW_RUBRIC.md`
6. `~/Char/datasets/ids/scripts/lookup_ids.py` (for SSEM structure mapping
   — this is mother-repo tooling, not the paper's official code).

Neither `~/Char/paper_reimpl/third_party/` nor
`github.com/Yaomingshuai/VQ-Font` was consulted. No WebFetch or WebSearch
call was issued.

## Decisions

### Architecture (Stage 0 — VQGAN)

1. **VQGAN codebook size = 1024**
   `[paper-cited Obsidian §訓練配置]`. Paper note explicitly states the
   1024-entry codebook on a 16×16 feature grid for 128 px input. We map this
   to `VQGANConfig.num_embeddings = 1024` and arrange `channel_mult =
   (1, 1, 2, 4)` so the encoder does 3 stride-2 downsamples (128 → 16).

2. **VectorQuantize uses straight-through gradient + commitment + codebook
   loss**, β = 0.25 `[paper-cited via VQ-VAE prior art]`. Van den Oord (2017)
   §3 — this is the canonical recipe the VQ-Font paper builds on without
   modification (the paper note does not name a custom quantizer).

3. **No adversarial discriminator at Phase 1**
   `[guessed-because-paper-vague]`. The note says "VQGAN-based" but does
   not specify discriminator depth, GAN loss weight, or LeCAM /
   spectral-norm regularisation. We default to L1-only reconstruction; the
   trainer exposes `recon_weight` / `vq_weight` knobs so a perceptual /
   adversarial term can be layered later without restructuring `train.py`.

4. **No perceptual loss (VGG / LPIPS) at Phase 1**
   `[guessed-because-paper-vague]`. Same rationale as decision 3 — paper
   note does not give the perceptual recipe.

5. **Encoder / decoder use 2 res-blocks per stage and a single attention
   block at the bottleneck** `[guessed-because-paper-vague]`. Paper note
   does not state widths or depths; defaults follow the taming-transformers
   shape that is the typical VQGAN baseline.

6. **z_channels = embed_dim = 256** `[guessed-because-paper-vague]`. Paper
   does not state the latent channel width. 256 is the taming-transformers
   default for the medium-size codebook.

### Architecture (Stage 1+ — Transformer + SSEM)

7. **Transformer cross-attention, 8 heads**
   `[paper-cited Obsidian §訓練配置]`. Note explicitly says
   "cross-attention 8 heads".

8. **Transformer depth = 6 blocks** `[guessed-because-paper-vague]`. Paper
   does not state depth; 6 is a small-scale default inherited from
   VQ-VAE-2 / DALL-E priors and small-Vit.

9. **3 reference characters** `[paper-cited Obsidian §訓練配置]`. Note says
   "cross-attention 3 reference characters". We accept R as a config and
   default to 3.

10. **MLP ratio = 4.0, dropout = 0.0** `[blind-impl-convention]`. Standard
    transformer recipe.

11. **Bidirectional attention (no causal mask)** `[guessed-because-paper-vague]`.
    Paper note says "Transformer predicts indices" without specifying
    whether decoding is autoregressive (causal) or in parallel (bidirectional).
    Parallel argmax decode is simpler and faster; if the official repo uses
    AR decoding, Phase 2 diff will surface that.

12. **SSEM = 14-way structure classifier** `[paper-cited 12 structures]` +
    `[blind-impl-convention]`. Paper specifies the 12 Chinese-character
    structure prior, mapped 1:1 to the `parse_structure` output of
    `~/Char/datasets/ids/scripts/lookup_ids.py` (left_right, top_bottom,
    left_mid_right, top_mid_bottom, surround_full,
    surround_open_{bottom,top,right,TR,TL,BR}, overlap). We add 2 sentinels:
    `atomic` (matches `parse_structure` fallback for atomic glyphs) and
    `unknown` (used at synthetic / TTF pretraining where no IDS data is
    available), bringing the head to 14 classes.

13. **SSEM loss form = cross-entropy on a structure head**
    `[guessed-because-paper-vague]`. Paper describes SSEM as "structure-level
    style matching" without giving a formula. CE was chosen for its
    minimality and direct compatibility with the existing structure-id
    label; a contrastive (same-structure positive) variant is listed in
    `paper_notes/04.md` open questions.

14. **SSEM injection = additive bias on every query token + prefix token
    in cross-attn context** `[guessed-because-paper-vague]`. Two injection
    surfaces give two gradient paths — the smoke test asserts both are
    grad-connected.

15. **`λ_struct = 0.1 → 0.2 → 0.3` across Stages A / B / C**
    `[guessed-because-paper-vague]`. Paper does not give a structure-loss
    weight. The ramp is designed so Stage A focuses on token CE,
    progressively trusting SSEM more as data quality / structure-id quality
    grows. All three are independently overridable in the YAMLs.

### Training schedule

16. **Stage 0: 200k iters, lr = 4e-5, batch = 32**
    `[paper-cited Obsidian §訓練配置]`. All three are paper-cited.

17. **Stage A: 300k iters, lr = 2e-4, batch = 32**
    `[paper-cited Obsidian §訓練配置]`. Note explicitly says "token
    refinement 300k iters" and lr 2e-4.

18. **Stages B / C iter counts and lrs are blind anneal defaults**
    `[guessed-because-paper-vague]`. Paper presents only the 200k + 300k
    counts; multi-stage fine-tune onto Ernantang is our addition. B = 50k @
    1e-4, C = 100k @ 5e-5 follow standard anneal-as-you-fine-tune practice.

19. **AdamW(β=(0.9, 0.999), weight_decay=0)** `[blind-impl-convention]`.
    Paper does not state. AdamW with no weight decay is the safe default
    for VQGAN / Transformer recipes.

20. **Gradient clip 1.0** `[blind-impl-convention]`. Insurance.

21. **No EMA on either VQGAN or Transformer** `[blind-impl-convention]`.
    Paper note does not require it. May be added in Phase 3 if Stage 0
    diverges.

### Implementation details (blind-impl convention)

22. **Frozen VQGAN at Stage 1+** `[paper-cited]`. Note "保留 codebook 與
    decoder 後段權重不變". We freeze the **entire** VQGAN (encoder +
    decoder + codebook) and verify via a smoke-test assertion that
    `requires_grad = False` for all VQGAN parameters and that the backward
    pass leaves them with `grad is None or 0`.

23. **Initial-synthesis stand-in = source content image at Phase 1**
    `[guessed-because-paper-vague]`. The paper's Stage 1 input is the
    output of an "any FFG synthesis module". We accept the
    source-content render as a stand-in until a real synthesis module
    (e.g. FontDiffuser Stage A checkpoint) is loaded via the
    `train_cfg.synthesis_ckpt` (TODO) field. This is acceptable for the
    plumbing smoke test but **must** be replaced before real Stage B/C runs.

24. **Learnable null token for padded ref slots**
    `[blind-impl-convention]`. When `ref_valid[i, r] = False` the slot's
    tokens are replaced with a per-token learnable null embedding so
    cross-attention keys never produce a fully-masked softmax row (which
    would NaN).

25. **GroupNorm groups probed for divisibility**
    `[blind-impl-convention]`. Same trick FontDiffuser uses — hard-coding
    `num_groups = 32` crashes when channels are awkward (e.g. 48 after a
    concat). We probe 32 → 16 → 8 → 4 → 2 → 1.

26. **Loaded VQGAN checkpoint via `weights_only=False`**
    `[blind-impl-convention]`. Stage A trainer falls back to random VQGAN
    init when the Stage 0 checkpoint is missing — print a `WARN` line but
    do not crash. This is what makes the entrypoint dry-run work without
    a pre-built Stage 0 ckpt.

## Things I would peek at GitHub for (if allowed)

The blind-impl constraint flagged these as the highest-value diffs to learn
from in Phase 2:

- **VQGAN encoder / decoder exact channel widths** (decision 5–6).
- **VQGAN adversarial discriminator and GAN loss weight** (decision 3).
- **Perceptual loss recipe** (decision 4).
- **Transformer depth and exact attention recipe** (decision 8, 11).
- **SSEM loss form (CE vs contrastive vs MSE)** (decision 13).
- **SSEM injection point — token bias vs prefix vs conditional LayerNorm**
  (decision 14).
- **`λ_struct` value (decision 15) and Stage B/C schedule (decision 18).**
- **Whether Transformer decodes AR or in parallel** (decision 11).
- **Synthesis-module choice** (decision 23).

## Open questions / known limitations

1. **`structure_id` labels in our manifests are not yet populated.** Phase 1
   reimpl reads `row['structure_id']` first, then falls back to
   `row['structure']` (string), then to 0 (unknown). The manifest-build
   pipeline in the mother repo needs a single-line addition to invoke
   `lookup_ids.parse_structure(row['char'])` before this code can make
   meaningful SSEM gradients on real data.
2. **`content` channel is single-bitmap.** Stages B/C can extend to multiple
   content channels via the existing `data_cfg.content_channels` list, but
   the model's initial-synthesis path currently expects 1-channel input
   matching `vqgan.in_channels`.
3. **No EMA on Transformer.** Standard but not implemented. Adding before
   Stage B handover is recommended.
4. **No synthesis-module checkpoint plumbing yet.** Stage 1+ uses the
   source content image as a stand-in.
5. **No discriminator for Stage 0.** As noted, paper-vague.

## Reproducibility checklist (per rubric)

- [x] `torch.manual_seed`, `numpy.random.seed`, `random.seed` set in
  `train._seed_everything` (called at the top of both `_run_vqgan_stage`
  and `_run_transformer_stage`).
- [x] No hardcoded SSH credentials or absolute internal paths in `.py`.
- [x] `--device` is honored from CLI; defaults to `cuda:0` at the entrypoint
  layer.
- [x] No silent `except: pass` in the package (verified via grep).
- [x] All data paths resolve through `paper_reimpl_shared.data.manifest`
  backend.
- [x] `tests/test_smoke.py` runs Stage 0 forward+backward+step, Stage 1+
  forward+backward+step (with frozen-VQGAN assertion), and sample
  (argmax + top-k) pipelines. CPU-only, no disk I/O.

## Verification commands run

From `papers/04_vq_font/`:

```
uv venv --python 3.11
uv pip install -e ../../shared
uv pip install -e .
uv pip install pytest ruff
uv run ruff check src/ tests/                  # All checks passed!
uv run pytest tests/test_smoke.py -x -v        # 3 passed in 0.92s

uv run python -m paper_reimpl_shared.runner.entrypoint \
    --paper vq_font --dry-run --synthetic --device cpu \
    --train src/vq_font/configs/train_stage_0_vqgan.yaml \
    --model src/vq_font/configs/model.yaml \
    --data src/vq_font/configs/data_stage_0_vqgan.yaml \
    --data-backend mac_symlink
# -> step=0 loss=1.2357 recon=1.0831 vq=0.1526; finite

uv run python -m paper_reimpl_shared.runner.entrypoint \
    --paper vq_font --dry-run --synthetic --device cpu \
    --train src/vq_font/configs/train_stage_a_ttf.yaml \
    --model src/vq_font/configs/model.yaml \
    --data src/vq_font/configs/data_stage_a.yaml \
    --data-backend mac_symlink
# -> step=0 loss=7.3820 token_ce=7.1104 struct_ce=2.7158 top1=0.34%; finite
# (Stage A loads a random VQGAN since no Stage 0 ckpt exists — expected
# for plumbing smoke; CE of ~7.1 ≈ log(1024) confirms uniform-init prior.)

uv lock
# -> Resolved 46 packages.
```
