# Blind Implementation — FontDiffuser (01)

Decision log for the Phase 1 blind reimplementation. Every non-trivial
choice is tagged either `[paper-cited §X]` when sourced from the Obsidian
paper note or Phase 0 spec table, or `[guessed-because-paper-vague]` when
we filled in a gap.

Tag `[blind-impl-convention]` means a tooling decision that is paper-neutral
but still worth recording (e.g. init schemes, dataloader plumbing). These
are not gating items.

## Source material consulted

1. `/Users/Ayueh/Documents/Obsidian Vault/research/papers/020_FontDiffuser單樣本擴散字體_AAAI2024.md`
2. `/Users/Ayueh/Documents/Obsidian Vault/research/papers/FontDiffuser_OneShot_Font_Generation_Denoising_Yang_2023.md`
3. `~/Char/paper_reimpl/reports/phase0_spec_table.md`, row 01
4. `~/Char/paper_reimpl/AGENTS.md`
5. `~/Char/paper_reimpl/shared/src/paper_reimpl_shared/...`
6. `~/Char/paper_reimpl/docs/REVIEW_RUBRIC.md`

Neither `~/Char/paper_reimpl/third_party/` nor `~/Char/FontDiffuser-main_0528/`
nor `github.com/yeungchenwa/FontDiffuser` was consulted. No WebFetch or
WebSearch call was issued.

## Decisions

### Architecture

1. **U-Net DDPM backbone** `[paper-cited Phase0 row 01]`. Paper note p.1
   "noise-to-denoise paradigm with U-Net". Shared
   `GaussianDiffusion` handles the forward / reverse process.

2. **Multi-scale Content Aggregation (MCA)** `[paper-cited Phase0 row 01]`.
   Note p.1 says MCA "aggregates global+local content features at multiple
   scales". My implementation fuses one content-encoder feature map into each
   U-Net stage that has matching channel width. The fusion operator is a
   **gated additive blend** with zero-init gate
   `[guessed-because-paper-vague]` — paper says nothing about the operator.

3. **Reference-Structure Interaction (RSI)** `[paper-cited Phase0 row 01]`.
   Note p.2 says "RSI cross-attention on reference". Implemented as
   multi-head cross-attention with U-Net spatial tokens as queries and style
   tokens (from StyleEncoder) as keys/values. Fires at every resolution in
   `attn_resolutions` plus the middle block `[guessed-because-paper-vague]`.

4. **Style Contrastive Refinement (SCR)** `[paper-cited Phase0 row 01]`.
   Note p.2 says SCR "uses a style extractor + contrastive loss with
   same-char-different-style negatives". Reimpl uses within-batch writer_id
   positives instead of structured "same-char-different-style"
   `[guessed-because-paper-vague]` — the paper's negative-mining scheme
   requires a batch sampler we do not yet have.

5. **One-shot conditioning** `[paper-cited Phase0 row 01]`. `ref_images[:, 0]`
   is used; additional slots are ignored. Multi-shot is not in scope.

6. **Channel mult `(1, 2, 4, 4)` and `base_channels=64`**
   `[guessed-because-paper-vague]`. Paper does not state widths. Choice
   follows OpenAI guided-diffusion 64×64-class for 128 px input.

7. **Attention at `16×16` only** `[guessed-because-paper-vague]`. Paper does
   not state attn resolutions. 16×16 is the de-facto DDPM default
   (Ho 2020, Improved DDPM).

8. **Style encoder = shallow 4-layer strided CNN** `[guessed-because-paper-vague]`.
   Paper does not say. We could plug in a VGG-19 or ResNet-50 trunk later
   (typical for style-transfer papers); shallow CNN keeps Phase 1 minimal
   and inspectable.

9. **`style_embed_dim=256`** `[guessed-because-paper-vague]`.

10. **Number of res blocks per stage = 2** `[guessed-because-paper-vague]`.
    DDPM default.

### Loss / training

11. **ε-prediction** `[guessed-because-paper-vague]`. Paper does not state
    target. ε is the DDPM default; we expose `prediction_target` in the train
    YAML so it can switch to x₀.

12. **Linear β schedule, β₁=1e-4, β_T=2e-2, T=1000** `[guessed-because-paper-vague]`.
    Paper says "DDPM" but does not pin the schedule. These are the DDPM 2020
    defaults; cosine schedule would be a reasonable alternative ablation.

13. **`λ_scr` ramp: 0.0 → 0.1 → 0.2** across stages A/B/C
    `[guessed-because-paper-vague]`. Paper does not give weight. The ramp is
    designed so Stage A is a clean denoiser warm-up, B introduces SCR while
    the model has competent backbone, C tightens style supervision.

14. **CFG dropout `p=0.1`** `[guessed-because-paper-vague]`. Ho & Salimans
    (2022) default. Paper note does not mention CFG.

15. **Learning rate `1e-4 → 5e-5 → 2e-5`** across stages
    `[guessed-because-paper-vague]`. Standard "anneal as you fine-tune"
    recipe.

16. **AdamW, β=(0.9, 0.999), weight_decay=0** `[blind-impl-convention]`. DDPM
    standard.

17. **Gradient clip 1.0** `[blind-impl-convention]`. Stable-diffusion default;
    DDPM does not require it but it's cheap insurance.

18. **`τ=0.1` in SCR contrastive** `[guessed-because-paper-vague]`. NT-Xent
    default (SimCLR).

19. **SCR uses predicted x₀, not predicted ε**. Paper note says "SCR
    supervises diffusion model" — we interpret this as supervising the model's
    sample (x₀-equivalent), not the noise estimate. `[guessed-because-paper-vague]`.

### Implementation details (blind-impl convention)

20. **MCA gate zero-init** `[blind-impl-convention]`. Starts as identity →
    learns to mix. Removes the risk of content path overwhelming the model
    before warm-up.

21. **Output conv non-zero-init** `[blind-impl-convention]`. OpenAI guided-diffusion
    zero-inits the final conv so step-0 prediction is exactly zero. I do not,
    because (a) it disconnects gradient on iter 0 in the content/style
    branches (the smoke test would crash) and (b) the effect washes out
    after a few optimizer steps.

22. **Learnable null-style token** `[blind-impl-convention]`. Replaces the
    style-encoder output when `ref_valid[:, 0]` is False. Keeps the
    cross-attention block well-defined under CFG dropout. Stable-diffusion
    uses a learnable null context vector for the same reason.

23. **GroupNorm group count probed for divisibility** `[blind-impl-convention]`.
    Hard-coding `num_groups=32` crashes on skip-concat widths like 48. We
    probe down 32 → 16 → ... → 1.

24. **Synthetic dataset routing** `[blind-impl-convention]`. When `--synthetic`
    or `data_cfg.source == 'synthetic'`, we use `SyntheticCalligraphyDataset`
    from the shared package. Real manifests resolve via the shared backend
    paths.

25. **Picklable `_CollateWithRefs`** `[blind-impl-convention]`. Closures
    over `max_refs` are not pickleable by `multiprocessing`, so the
    `DataLoader` collate function is a module-level callable.

### Sampler

26. **DDPM and DDIM both supported** `[paper-cited Phase0 row 01]` (DDPM) +
    `[blind-impl-convention]` (DDIM offered as a faster qualitative
    inspection sampler).

27. **CFG between conditional and null-ref-unconditional**
    `[guessed-because-paper-vague]`. Paper note does not mention CFG; this is
    a sensible default for one-shot image conditioning.

## Things I would peek at GitHub for (if allowed)

The blind-impl constraint flagged these as the highest-value diffs to learn
from in Phase 2:

- **Exact channel widths and depth** of the U-Net trunk (decisions 6, 9, 10).
- **MCA fusion operator**: gated-add vs concat-then-conv vs attention-gated
  (decision 2).
- **RSI placement**: bottleneck-only vs every-resolution (decision 3).
- **Style encoder backbone**: shallow CNN vs pretrained VGG/ResNet
  (decision 8).
- **SCR negative-mining**: within-batch writer_id (ours) vs structured
  same-char-different-style (paper) (decision 4).
- **β schedule, T, λ_scr, τ, p_drop, lr schedule** (decisions 11–15, 18).
- **Whether the official model uses EMA** (we currently do not).
- **Whether the official sampler is plain DDPM or accelerated (DDIM / PNDM)**.

## Open questions / known limitations

1. The SCR style extractor is built but not pretrained. The full pipeline
   expects a Stage 0 writer-id classification pretrain that we do not
   implement at Phase 1.
2. Cross-language (CN→KR) eval — out of Phase 1 scope.
3. Real Stage B / C training has not been launched; the configs point at
   smoke manifests so dry-run does not require the full manifest snapshot.
4. The current `compute_loss` recomputes embeddings on the predicted x₀
   every step. For large batches at high resolution this doubles the style
   extractor cost; an EMA-of-extractor or queue-based memory bank would amortise
   it.
5. No EMA on the diffusion model itself. DDPM-typical EMA decay 0.9999
   should be added before Stage B handover.
6. Content channels currently hardcoded to `[bitmap]`. The mother repo's
   content cache supports multi-channel (bitmap / SDF / skeleton / ...) — we
   left the YAML flexible (`content_channels: [bitmap, sdf, ...]`) but the
   smoke run uses bitmap only.

## Reproducibility checklist (per rubric)

- [x] `torch.manual_seed`, `numpy.random.seed`, `random.seed` set in `train.main`.
- [x] No hardcoded SSH credentials or absolute internal paths in `.py`.
- [x] `--device` is honored, defaults to `cuda:0`, not hardcoded.
- [x] No silent `except: pass` in the package.
- [x] All paths resolve through `paper_reimpl_shared.data.manifest`.
- [x] `tests/test_smoke.py` runs forward + backward + 1 optimizer step + 1
      sample step on CPU.

## Verification commands run

From `papers/01_fontdiffuser/`:

```
uv venv --python 3.11
uv pip install -e ../../shared
uv pip install -e .
uv pip install pytest
uv run pytest tests/test_smoke.py -x -v
# -> 2 passed in 0.91s

uv run python -m paper_reimpl_shared.runner.entrypoint \
    --paper fontdiffuser --dry-run --synthetic --device cpu \
    --train src/fontdiffuser/configs/train_stage_a_ttf.yaml \
    --model src/fontdiffuser/configs/model.yaml \
    --data src/fontdiffuser/configs/data_stage_a.yaml \
    --data-backend mac_symlink
# -> 1 step, loss_simple=1.0814, loss_scr=0.0000, finite.

uv lock
# -> 45 packages resolved.
```
