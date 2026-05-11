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

## Phase 2 corrections (2026-05-11)

Cloned `yeungchenwa/FontDiffuser` @ `7b28ce9c3b357f4fb23296622f458cf169803539`
into `third_party/01_fontdiffuser/` and addressed the 5 P0 deltas flagged
in `reports/github_diff.md`. Files touched (see `Returned summary` at the
bottom of the implementation report for line ranges):

1. **RSI: deformable-conv warp + offset L1 loss** —
   replaces the plain cross-attention `RSIBlock` on the up path.
   - New module: `src/fontdiffuser/model.py::OffsetRefStrucInter`.
   - Added `style_content_encoder` (a second `ContentEncoder` running on
     the style image) so the offset head has the per-stage style-content
     feature pyramid as conditioning context.
   - U-Net forward now returns `(eps_pred, offset_l1_sum)`, but the top-
     level `FontDiffuser.forward` keeps the single-tensor contract by
     stashing the sum in `self._last_offset_l1` (preserves compatibility
     with the shared `GaussianDiffusion._model_pred`).
   - `FontDiffuserConfig.offset_l1_weight: float = 0.5` (matches official
     `offset_coefficient`).

2. **SCR: pretrained VGG-16 + 6 projector heads + InfoNCE(τ=0.07)** —
   replaces the writer_id supervised NT-Xent.
   - New module: `src/fontdiffuser/model.py::SCRModule` (with helpers
     `_SCRStyleFeatExtractor` and `_SCRProjector`).
   - Uses `torchvision.models.vgg16(pretrained=True)` (frozen) split at
     the 6 MaxPool boundaries. Each stage's GAP+GMP pooled features
     compress through a 1×1 conv, then a 3-layer MLP projector
     (`stage_C → 1024 → 2048 → 2048`) with L2 normalisation.
   - Positive augmentation: `kornia.augmentation.RandomResizedCrop(
     scale=(0.8,1.0), ratio=(0.75,1.33))` on the target image.
   - Negative sampling: dataset emits `batch["neg_images"]` of shape
     `[B, num_neg=16, C, H, W]` (still TODO at the dataset layer — see
     "Outstanding" below).
   - InfoNCE: `info-nce-pytorch::InfoNCE(temperature=0.07,
     negative_mode='paired')` averaged across the requested
     `nce_layers` (default `(0, 1, 2, 3)`). Falls back to a manual
     paired InfoNCE if the `info_nce` package is missing.
   - Back-compat: `StyleExtractor = SCRModule` so older callers keep
     working.

3. **Phase 1 loss = MSE + 0.01·VGG-Perceptual + 0.5·Offset** —
   replaces the single-term diffusion loss.
   - New module: `src/fontdiffuser/model.py::ContentPerceptualLoss`
     (VGG-16 enc_1/enc_2/enc_3 MSE on re-normalised inputs).
   - `compute_loss` in `src/fontdiffuser/train.py` now accepts
     `perceptual_loss_fn`, `scr_module`, `perceptual_weight`, and
     `offset_l1_weight`. `main()` builds the perceptual + SCR heads
     when their coefficients are positive.

4. **CFG dropout drops both content and style** —
   replaces "ref-only" dropout.
   - `compute_loss` zeros `content` and nulls `ref_valid` on the same
     batch rows with probability `cfg_drop_prob`.
   - `sample.py::sample` now passes `cfg_uncond_drops_content=True` to
     match (the shared sampler zeros content on the uncond branch).

5. **MCA: concat + SE at inner two stages only** —
   replaces gated-add at every stage.
   - Rewrote `MCAFuse` as
     `Conv1x1(c) → concat → GN-SiLU-Conv1x1 → SELayer(reduction=32) →
     +concat → GN-SiLU-Conv1x1 → out`.
   - `FontDiffuserConfig.resolved_mca_stages()` returns `(1, …, N-2)` by
     default — inner two stages on a 4-stage U-Net.
   - Token-level style cross-attention (`RSIBlock` on style tokens)
     fires at MCA stages on both down and up paths plus the middle.

### Dependencies

`pyproject.toml` now requires:

- `torchvision>=0.16` — `DeformConv2d` (RSI) and pretrained VGG-16
  (perceptual + SCR).
- `kornia>=0.7` — `RandomResizedCrop` positive augmentation.
- `info-nce-pytorch>=0.1.4` — `InfoNCE(temperature=0.07,
  negative_mode='paired')`.

### Outstanding (deferred — *not* P0)

- **Dataset-level `num_neg=16` negative sampler.** The training loop is
  wired but the dataset adapter still produces no `neg_images`. Phase 2
  Stage B/C training will be no-op on SCR until
  `FontDiffuserPairDataset` (or a new class) starts emitting the
  same-content-different-style negatives. See
  `third_party/01_fontdiffuser/dataset/font_dataset.py:78-99` for the
  recipe.
- **SCR ckpt pretraining.** Paper expects ~210k steps of SCR self-training
  before Phase 2. `train_cfg["scr_ckpt_path"]` is consumed when set, but
  we do not yet have a training script for the SCR module itself.
- **Up/down style cross-attention on every MCA stage.** The blind impl
  fired RSI only at `attn_resolutions`. We now fire token-level style
  cross-attention at every MCA stage; this is a behavioural change worth
  documenting in any retrospective ablation.
- **β `scaled_linear` schedule, DPM-Solver++, weight_decay=1e-2,
  resolution=96, channel widths (64,128,256,512), RGB in/out** — all
  P1 items in `reports/github_diff.md`, not addressed in this Phase 2
  patch.

### Verification (Phase 2)

```bash
uv pip install -e .  # picks up torchvision/kornia/info-nce-pytorch
uv run ruff check src/ tests/
# -> All checks passed!

uv run pytest tests/test_smoke.py -x -v
# -> 3 passed (added test_smoke_perceptual_offset_contribute)

uv run python -m paper_reimpl_shared.runner.entrypoint \
    --paper fontdiffuser --dry-run --synthetic --device cpu \
    --train src/fontdiffuser/configs/train_stage_a_ttf.yaml \
    --model src/fontdiffuser/configs/model.yaml \
    --data src/fontdiffuser/configs/data_stage_a.yaml \
    --data-backend mac_symlink
# -> step=0 loss_total=1.9884 loss_simple=1.1866 loss_perc=80.10
#    loss_offset=0.0016 loss_scr=0.0000  (all finite)
```

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
