# DL Review — 08_dp_font — Gate 1 (blind reimpl, pre-Phase-2)

## Verdict: PASS-WITH-NITS

Blind-impl review. No peek at official `third_party/` or upstream GitHub
performed; the paper has `facts_code_url: null` so no upstream existed to
peek even if allowed. **NOT CONTAMINATED.**

The implementation matches every load-bearing claim in `paper_notes/08.md`
that can be verified against code at this stage. The single review gap
flagged below (PDE form) is documented honestly in `blind_impl.md` as
`[guessed-because-paper-vague]` and is out of scope for Gate 1.

## Checked (rubric items)

### Loss correctness
- [✓] **Loss formula matches paper note §3** — `L_total = L_simple + λ_PINN * L_PINN`
  - `train.py:111` `loss_simple = F.mse_loss(model_pred, diff_batch.target, reduction="mean")` matches DDPM ε-prediction MSE.
  - `train.py:136-146` aggregates `L_PINN` (three sub-terms) with master weight `pinn_weight`. Tracks `paper_notes/08.md` §3 exactly.
- [✓] **Reduction mode `mean`** consistent across `loss_simple` (`train.py:111`) and all three PINN terms (`pinn_losses.py:169, 206, 232`).
- [✓] **Weight combination matches blind-impl decisions 12–14**: per-term weights in `pinn_loss()` (`pinn_losses.py:267-271`) and stage-wise master `pinn_weight` in yamls (0.0 → 0.05 → 0.1). Paper publishes no values, so the [guessed] tag is honest.
- [✓] **Diffusion target = epsilon** (configured in all three train yamls). Schedule alignment: training and sampling both flow through `GaussianDiffusion.predict_x0`, no x0/ε crossover.
- [✓] **CFG dropout p = 0.1, per-attribute Bernoulli**: `train.py:45-55` `_cfg_drop` + `train.py:92-94` applies independently to writer/script/char. Matches Ho & Salimans 2022 default; rubric line 22 satisfied.

### Gradient flow (rubric "Conditioning paths")
- [✓] **PINN backprops into U-Net — author's own test passes.** `test_pinn_contributes_to_unet_gradient` (`tests/test_smoke.py:160-201`) toggles `pinn_weight` 0 → 10 with identical seeds + state-dict copy, asserts `|grad_b - grad_a| > 1e-6`. Verified by `uv run pytest -x -v` → all 5 tests green. This is the DP-Font-specific must-have from the rubric ("DP-Font: PINN 物理 loss 真有反向傳回 generator") and it is correctly verified.
- [✓] **Each PINN sub-term is differentiable**: `test_pinn_loss_is_differentiable` (`tests/test_smoke.py:132-157`) — green.
- [✓] **No spurious `.detach()` on critical path.** Only detaches found:
  - `pinn_losses.py:165` `bg_mask.detach()` and `pinn_losses.py:204` `stroke_mask.detach()` — **correct**: detaching the *mask* prevents the degenerate solution "mask everything out". Penalty signal still flows through `lap` / `sig` which are computed without detach. Standard PINN practice.
  - `train.py:115,145,148` `.detach().cpu()` only used for logging scalars.
  - `pinn_losses.py:273-275` `.detach()` for log dict only.
- [✓] **All three branches receive gradient** (`test_smoke_forward_backward_step` lines 108-124): content_encoder, guidance, unet each get non-zero grad.
- [✓] **Multi-attribute path connected**: `MultiAttributeGuidance.forward` (`model.py:205-268`) sums embeddings into `h`, then `self.fuse(h)`; `ResBlock.forward` (`model.py:340`) adds `cond_proj(cond)` to every block. Conditioning vector reaches every ResBlock in down/mid/up trunks (`model.py:524-525, 534, 536, 543-544`). Wiring matches the rubric requirement "multi-attribute FiLM in every ResBlock".
- [✓] **Stroke-order conditioning verified active**: token + positional embed + masked-mean-pool (`model.py:235-247`); padding `-1` masked via `valid = so >= 0`. Pooled vector is *added* to `h` (line 247) so it flows into `self.fuse` and then into every ResBlock through `cond`.
- [✓] **Per-attribute CFG dropout**: `train.py:45-55` flips writer/script/char to the embedding's null slot (`vocab_size`) independently. Embedding tables have `vocab_size + 1` entries (`model.py:167-169`) reserving the trailing id as the learnable null.
- [N/A] **EMA / zero-init**: paper note does not require either. `ContentFuse.gate` IS zero-init'd (`model.py:394-395`) — a sensible safety so the content stream contributes nothing on step 0 and ramps via the gate.

### Schedule & sampler
- [✓] **Cosine β, T=1000** (`configs/train_stage_*.yaml`); `α_T ≈ 0` guaranteed by Improved-DDPM cosine impl in `shared/.../gaussian.py:26-40`.
- [✓] **Sampler matches noise convention**: epsilon prediction throughout; `predict_x0` (`shared/.../gaussian.py:126-132`) handles the ε→x0 inversion identically for training (PINN x0 reconstruction) and inference.
- [✓] **Time embedding**: sin/cos + 2-layer MLP (`model.py:128-141, 422-426`); width matches `cfg.time_embed_dim`.
- [✓] **DP-Font-specific sample path**: `sample.py:_FrozenCondAdapter` correctly bakes `stroke_order/ink_intensity/font_size` so the shared sampler (which doesn't know about these) routes them on every call. `cfg_uncond_drops_content=False` (`sample.py:108`) — matches training-time semantics (DP-Font drops categorical attrs, NOT content).

### Data normalization
- [✓] **Image range `[-1, 1]`** consistent across PINN tensor convention (`pinn_losses.py:55-58`), prediction clamp (`shared/.../gaussian.py:128, 132`), and dataset (legacy loader).
- [✓] **No horizontal flip / large rotation** — DP-Font dataset reuses the shared `CalligraphyJsonlDataset`/`SyntheticCalligraphyDataset`; neither flips. Calligraphy correctness preserved.
- [✓] **Content cache alignment**: Stage A `[bitmap]` (`data_stage_a.yaml:18`), Stage B/C `[bitmap, skeleton]` (`data_stage_b.yaml:8`). `skeleton_channel_index: 1` in `train_stage_b/c.yaml` matches the channel order. `train.py:128-133` correctly slices the skeleton channel for the nib-motion term.

### PINN physics sanity (Gate-1 special focus)
- [✓] **Ink-diffusion residual = `(ν ∇²I)² * bg_mask`** (`pinn_losses.py:138-169`).
  - Steady-state diffusion `ν ∇²I + s = 0` with `s ≈ 0` in background ⇒ minimise `|ν ∇²I|²` on bg. Mathematically sound.
  - Laplacian via 3x3 isotropic kernel with replicate padding (`pinn_losses.py:81-108`). Standard.
  - bg_mask: `sigmoid(-10(I - τ))` with `τ=0` — smooth soft mask, differentiable through `I`.
- [✓] **Nib-motion smoothness = `|∇²(skeleton or x0)| * stroke_mask`** (`pinn_losses.py:172-206`).
  - Penalises Laplacian L1 inside ink region — physically motivated as "no abrupt brush direction reversals".
  - Falls back to `x0_pred` when no skeleton channel (Stage A path) — handled correctly: `skeleton=None` ⇒ `sig = x0_pred` (line 197-198).
  - Bilinear resize when `skeleton.shape != x0_pred.shape` (line 200-201) — defensive.
- [✓] **Stroke-continuity speckle penalty** (`pinn_losses.py:209-232`).
  - `neighbour_avg = (9·box_blur - center) / 8` correctly computes 8-neighbour mean excluding self. Math checked: box kernel is `1/9 * sum_{3x3}`, so `9·blur = sum_{3x3}`; subtracting `x0_pred` (the centre) and dividing by 8 gives the 8-neighbour mean. Algebra clean.
  - `speckle = ink * relu(-neighbour_avg)` — ink pixel surrounded by background (negative neighbour_avg → `relu(-neighbour_avg)` is positive). Sane.
- [✓] **PINN evaluated on predicted x0, not ε**: `train.py:124` `x0_pred = diffusion.predict_x0(x_t, t, model_pred)`. Required because physical priors are on ink density, which corresponds to x0. The closed-form ε→x0 inversion (`gaussian.py:126-132`) propagates gradient through `model_pred`, so the chain `eps_pred → x0_pred → L_PINN` is unbroken.

### Training dynamics
- [✓] **Smoke test green** — `uv run pytest -x -v` → 5/5 PASSED in 1.45s. No NaN, all params finite after optimiser step.
- [✓] **Grad clip 1.0** wired (`train.py:293-294`) and set in all three stage yamls.
- [✓] **AdamW β=(0.9, 0.999), weight_decay=0** (`train.py:254-259`) — matches blind-impl decision 20.
- [✓] **Seeding**: `_seed_everything` (`train.py:157-162`) sets random/numpy/torch + CUDA seeds. Rubric line 99 satisfied.
- [N/A] **batch×grad_accum scaling**: DP-Font's paper does not give a target effective batch; blind-impl chose bs=16 for single 3090. No grad-accum used (single forward/backward).

### Stroke_order placeholder (rubric explicitly marks non-blocking)
- [✓] `dataset.py:48-72` `synthesise_stroke_order` — deterministic SHA256-based placeholder. Documented in module docstring (`dataset.py:1-15`), `paper_notes/08.md` §8, and `blind_impl.md` decision 25 + open-question 1. Stage B/C launch instructions clearly say to swap before real training. **Non-blocking for Gate 1.**

## Nice-to-have (PASS-WITH-NITS backlog)

1. **Docstring vs code mismatch — additive FiLM, not AdaLN scale-shift.**
   `model.py:28-29` claims "AdaLN-style scale-shift modulates U-Net feature
   maps", but `ResBlock.forward` (`model.py:337-342`) only adds time/cond
   projections to the residual (`h = h + time_proj(...) + cond_proj(...)`). No
   `gamma * norm + beta` split. This IS valid FiLM (Perez 2018 §3.1, the
   "scale=1, only shift" variant), but the AdaLN claim is misleading.
   *Fix*: either change the docstring at `model.py:28` to "additive FiLM-style
   bias" or upgrade to true AdaLN-Zero (split `cond_proj` into γ/β, apply post-
   norm). Phase 2 candidate (decision 7 in `blind_impl.md` already flags
   AdaLN-Zero as a potential upgrade).

2. **Stroke-embedding has one unused slot.**
   `model.py:170` `nn.Embedding(stroke_vocab_size + 2, d)` — comment says "+1
   for [PAD], +1 for [SOS]" but only the `vocab_size + 1` slot is ever indexed
   (as padding null, line 238). Index `vocab_size` is allocated but unused.
   *Fix*: change to `+ 1` or actually use the second slot for an [SOS] token at
   the start of the sequence. Cosmetic, no correctness impact.

3. **`predict_x0` clamp may zero PINN gradient when x0 saturates.**
   `shared/.../gaussian.py:128, 132` clamps predicted x0 to [-1, 1]. Once a
   pixel saturates, its grad through clamp is 0, so L_PINN no longer pushes
   that pixel. Early-training prediction is far from saturation so the
   author's toggle test (PINN truly backprops) still passes. *Fix-if-tight*:
   pass `clamp=False` from PINN call sites, or use `tanh` instead of clamp.
   Not required for Gate 1 — author's test already shows non-zero grad
   contribution at init.

4. **Ink-intensity / font-size placeholders should be flagged in train log.**
   `dataset.py:75-78` synthesises both scalars from SHA256. Stage B/C runs
   will silently consume them as if real. *Fix*: print a one-line WARNING in
   `train.py:main` when `source == manifest` AND no real `ink_intensity`/
   `font_size` columns are detected, so launch-time logs make the placeholder
   status visible.

5. **No EMA on diffusion weights** — `blind_impl.md` open-question 3
   explicitly flags this as a Stage B prerequisite. Add 0.9999-decay EMA
   before Stage B handover, not before Gate 1.

## Suggested ablations (optional, Phase 2)

- **PINN component ablation** (`weight_diffusion`, `weight_nib`,
  `weight_continuity` each individually 0): isolates which physical prior
  matters most. The paper does not report this.
- **`λ_PINN` sweep at Stage C** (0, 0.05, 0.1, 0.5, 1.0): characterises the
  trade-off between FID and physical plausibility. Blind-impl decision 14
  picks 0.1 with no anchor — this is the most defensible ablation to run.
- **Additive FiLM vs AdaLN-Zero**: directly tests nit #1 above. Cheap A/B,
  one config flip.
- **Stroke-order encoder swap**: mean-pool (current) vs tiny Transformer +
  cross-attention into U-Net. Blind-impl decision 8 flags this as the
  highest-value architectural unknown.

## Files inspected (absolute paths)

- `/Users/Ayueh/Char/paper_reimpl/papers/08_dp_font/paper_notes/08.md`
- `/Users/Ayueh/Char/paper_reimpl/papers/08_dp_font/reports/blind_impl.md`
- `/Users/Ayueh/Char/paper_reimpl/papers/08_dp_font/src/dp_font/model.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/08_dp_font/src/dp_font/pinn_losses.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/08_dp_font/src/dp_font/train.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/08_dp_font/src/dp_font/dataset.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/08_dp_font/src/dp_font/sample.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/08_dp_font/src/dp_font/configs/{model,train_stage_*,data_stage_*}.yaml`
- `/Users/Ayueh/Char/paper_reimpl/papers/08_dp_font/tests/test_smoke.py`
- `/Users/Ayueh/Char/paper_reimpl/shared/src/paper_reimpl_shared/diffusion/gaussian.py` (for predict_x0 / sampler interface check)

## Verification commands run

```
uv run pytest tests/test_smoke.py -x -v
  → 5 passed in 1.45s
  → test_pinn_loss_is_differentiable PASSED  (each sub-term has grad on x0_pred)
  → test_pinn_contributes_to_unet_gradient PASSED  (λ_PINN 0→10 changes U-Net grad norm)
  → test_smoke_forward_backward_step PASSED  (content/guidance/unet branches all receive grad)

grep -nE "\.detach\(|no_grad" src/dp_font/*.py
  → only intentional sites (mask-detach in PINN, logging detach, sample.py no_grad). No critical-path grad-killer.
```
