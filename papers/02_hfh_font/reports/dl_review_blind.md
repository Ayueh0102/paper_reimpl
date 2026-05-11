# DL Review — 02_hfh_font — Gate 1 (post blind impl)

## Verdict: PASS-WITH-NITS

The blind impl is internally consistent, the smoke contract passes, and the four
"big claims" (latent diffusion + component cross-attn + SDS distillation +
style-guided SR) are all wired up to forward/backward. No FAIL items that
block Phase-2. All five author self-flagged weaknesses are valid and tracked
below; one additional latent bug found (SDS apples-vs-oranges under
`epsilon` prediction target), three nits worth fixing before any real run.

No contamination detected — the worker explicitly tagged every interpolation
in `reports/blind_impl.md`, and `paper_notes/02.md` independently lists the
same `[guessed]` items. Authorial intent is auditable.

---

## Checked (rubric items)

### Loss correctness
- [x] **loss formula matches paper §4.1** — `L_simple = MSE(pred, target)`
  with `target = z0` when `prediction_target="x0"` (Plan-A default), or `ε`
  when `"epsilon"`. Implemented at `src/hfh_font/model.py:632` against
  `shared/diffusion/gaussian.py:117` (target switch).
- [x] **reduction mode** — `F.mse_loss(...)` defaults to `mean`, matching
  per-sample average expectation; lr=1e-4 is calibrated for mean reduction.
- [x] **weights** — Phase-1 is single-term `L_simple`; no multi-term mixing
  yet, so no weight inconsistency to flag. SR has L1-only (paper says
  "style-guided SR" but does not specify weights — author flagged the
  LPIPS gap explicitly).
- [x] **diffusion target** — `prediction_target="x0"` in both
  `configs/model.yaml:23` and shared `GaussianDiffusion`. Schedule aligned:
  `q_sample` uses `√ᾱ_t·x0 + √(1-ᾱ_t)·ε` (`gaussian.py:120`).
- [x] **CFG dropout** — applied in
  `src/hfh_font/model.py:614-620`. See **nit C-1** for the over-broad
  channel coverage (paper specifies p̂=0.1 on style refs only).

### Gradient flow
- [x] **no detach on critical path of compute_loss** — only the VAE encode is
  wrapped in `torch.no_grad()` (`model.py:605`), matching the paper's
  "VAE frozen during diffusion training" claim. ComponentEncoder is
  trainable and on-path.
- [x] **conditioning path connectivity** — `_build_cond` (line 405) sums
  `t_emb + char_emb + writer_emb + script_emb` into the AdaLN-Zero cond
  vector that drives every `_ResBlock`'s `_AdaLNZero` modulation (lines
  266, 274). Cross-attention K/V come from `ComponentEncoder(refs)`
  (`model.py:575`), not from target. Confirmed against rubric:
  *"HFH-Font: component attention keys/values from reference glyph, not
  target"* — OK.
- [x] **AdaLN-Zero init** — `nn.init.zeros_(self.proj.weight/bias)` at
  `_AdaLNZero.__init__` (line 253-254). Output conv is also zero-init
  (line 393-394). Identity-at-init pattern. See nit D-1 about ComponentEncoder
  receiving zero gradient through cross-attn at the *very first* step.
- [x] **grad clip** — `clip_grad_norm_(parameters, 1.0)` honored in
  `train.py:171-173` when `grad_clip > 0` (configured at 1.0 in all stage
  yamls).

### Schedule & Sampler
- [x] **β schedule** — `build_beta_schedule` supports `linear` (default) and
  `cosine`. Linear with β1=1e-4, βT=2e-2 matches DDPM canonical default
  (paper note doesn't specify, [guessed-defensible]).
- [x] **sampler matches noise convention** — `predict_x0` (`gaussian.py:126`)
  branches on `prediction_target`. `p_sample` / `p_sample_ddim` both go
  through `predict_x0` → posterior mean → next x_t. Consistent.
- [x] **time embedding** — `_sinusoidal_time_embedding` builds the standard
  half-cos/half-sin embedding (note: cos-first vs DDPM sin-first; the
  following MLP absorbs the permutation, harmless).
- [x] **train/infer step disparity documented** — paper §訓練配置 specifies
  "DDPM 10 steps trailing" inference vs T=1000 training. Blind impl uses
  DDIM-10 as the "close-enough" Phase-1 substitute, flagged in
  `blind_impl.md` weakness #4.

### Data normalization
- [x] **image range [-1, 1]** — `load_grayscale_tensor` returns
  `array / 255.0 * 2.0 - 1.0` (`legacy.py:64-65`). VAE decoder ends in
  `nn.Tanh()` (`model.py:139`). Symmetric.
- [x] **no horizontal flip / rotation augment in calligraphy path** —
  grep of `legacy.py` for `flip|rotat|RandomAffine|transform` returns no
  matches. Compliant with rubric "書法不能水平翻轉、不能大幅 rotate".
- [x] **content cache aligned** — `[bitmap, sdf, skeleton]` axis order is the
  project-wide convention; configured in `data_stage_a.yaml:4` and threaded
  through `ManifestNotFoundError` fallback.

### Conditioning paths (HFH-Font specialization)
- [x] **K/V from reference glyphs, not target** — confirmed at
  `HFHFontModel.forward:575` (`ref_tokens = self.component_encoder(ref_images, ref_valid)`)
  and `_CrossAttention.forward:292-306` (k/v derived from `tokens`).
  Target is *only* `z_t` (corrupted latent) on the q side.
- [~] **component-aware encoding** — see **F-flagged #2** below; the encoder
  is a vanilla CNN + adaptive pool, no IDS or radical-aware front-end.
  Author self-flagged, conforms to "blind reimpl" honest contract.

### Training dynamics
- [x] **loss finite over 1 forward+backward+step+forward2** — verified by
  `tests/test_smoke.py::test_smoke_build_and_train_step` (asserts at
  lines 84, 95). Tests pass per Phase-1 contract.
- [n/a] **EMA** — paper note does not mention EMA. None implemented.
  Standard but not required; flag as nice-to-have.
- [x] **batch / lr scaling** — paper batch 64–128, lr unspecified. Blind
  impl batch 16 with lr 1e-4 — within standard diffusion ballpark. Linear
  rule (lr_paper * batch_ours / batch_paper) would suggest lr ≈
  1.6e-5–3e-5. lr=1e-4 is on the high end but defensible at small batch;
  flag as nit.

---

## Confirmed self-flagged weaknesses (PASS-WITH-NITS, route to Phase-2)

| # | Author claim | Status | Route |
|---|---|---|---|
| 1 | TinyVAE not pretrained, random init | **CONFIRMED**. `model.py:519-524` constructs `TinyVAE` from scratch. `train.py` has no `vae_ckpt` loader. Latent diffusion on a random-init VAE will not converge to glyph-quality reconstruction. | Phase-2 must add VAE pretrain stage or load `stable-diffusion` VAE / Plan-A shared VAE before Stage A diffusion training. |
| 2 | ComponentEncoder has no IDS plumbing | **CONFIRMED**. `ComponentEncoder` (`model.py:170-226`) is a stem→backbone→adaptive pool stack; the "component" interpretation is just a `√K × √K` grid pool. The phrase "component" is structural-claim only — there is no IDS lookup, no parsing-aware bias. | Phase-2 follow-up: integrate `~/Char/datasets/ids/` (CNS11643 + zispace) per-ref decomposition and emit per-component tokens with positional ids. |
| 3 | SDS loss is MSE placeholder | **CONFIRMED + ADDITIONAL BUG**. `compute_sds_loss` (`model.py:638-697`) computes `MSE(z0_student, teacher_x0.detach())` with `torch.no_grad()` around the teacher and `z0_student.detach()` fed into `q_sample`. This severs the SDS chain — what remains is plain student↔teacher distillation MSE, not the SDS gradient ∇θ E[w(t)(ε_T − ε)∂z_t/∂θ]. **Additional bug not flagged by author**: when `prediction_target="epsilon"`, line 668 `z0_student` actually holds ε prediction, but line 696 still compares it to `teacher_x0` — apples-vs-oranges. Currently masked because all configs default to `x0`. | Phase-2 must (a) switch loss to actual SDS gradient form or (b) explicitly rename to "ProxyDistillationLoss" and decide whether to support `epsilon` target (add a target-aware branch). |
| 4 | DDIM 10-step substitute for trailing-DDPM | **CONFIRMED**. `sample.py:39` calls `diffusion.sample(..., sampler="ddim")`, which uses uniform step decrement (`shared/gaussian.py:379`), not the "trailing 10 timesteps of the schedule" the paper describes. FID-relevant. | Phase-2: implement trailing timestep selector or document the substitution in the headline-number caveat. |
| 5 | CFG dropout applied uniformly to all cond channels | **CONFIRMED**. `model.py:614-620` drops `char_id / writer_id / script_id / refs` each at p=0.1. Paper §訓練配置 specifies p̂=0.1 on **style-ref→same-char replacement in the last 10% of iters** only. Dropping `char_id` is especially bad — char is the *content* signal; dropping it teaches the network to ignore the target glyph identity. | Phase-2: split dropouts (`cfg_dropout_refs: 0.1`, `cfg_dropout_writer: 0.1`, `cfg_dropout_char: 0.0`) and only enable in the last 10% of iters per paper. |

---

## Additional findings (not on author self-flag list)

### F-flagged (FAIL-on-paper-fidelity, not blocking smoke / Gate 1)

None. All paper-fidelity gaps are already on the self-flag list or
documented in `blind_impl.md` as `[guessed]`. Honest blind contract.

### N-flagged (nits — fix before Stage A real run)

**N-1: SDS loss latent target-mode bug (already noted under self-flag #3)** —
`src/hfh_font/model.py:668-696`. When `prediction_target="epsilon"`,
`z0_student` is the ε prediction, but line 696 `F.mse_loss(z0_student,
teacher_x0)` compares ε vs x0. Fix: convert student output to x0 via
`diffusion.predict_x0(z_T, t_max, z0_student)` before MSE, *or* assert
`diffusion.prediction_target == "x0"` at function entry.

**N-2: No checkpoint loading in `train.py`** — `train.py` builds model fresh
every invocation; there is no `--resume` / `model_cfg.init_from` /
`vae_ckpt` plumbing. Stage B and Stage C YAMLs reference reduced lrs that
only make sense after Stage A weights are loaded, but the current scaffold
will start Stage B/C from a random init. Fix: add `init_from: str | None`
to train_cfg and a small `_maybe_load_ckpt(model, path)` helper in
`train.py` before the `optimizer = _build_optimizer(...)` line. Critical
before any multi-stage real run.

**N-3: `components_per_ref` requested vs actual mismatch** —
`ComponentEncoder.__init__:204-207` overrides the requested value to
`side * side` where `side = round(sqrt(K))`. For configured `K=8`,
`side=3`, actual `K=9`. For configured `K=4`, `side=2`, actual `K=4`
(matches). Either change to `floor(sqrt(K))**2` with a doc note, or
support non-square grids; current behavior silently changes the requested
token count.

**N-4: Cross-attention zero-init means ComponentEncoder gets zero gradient
on the very first step** — `_CrossAttention.out_proj` is zero-init
(`model.py:289-290`). Standard AdaLN-Zero / DiT pattern, will self-correct
within the first few steps as `out_proj.weight` becomes non-zero. Worth
adding an explicit smoke assertion `any(p.grad.abs().sum() > 0 for p in
model.component_encoder.parameters())` *after at least 2 steps* to
guarantee the path stays connected as the codebase grows.

**N-5: `_sinusoidal_time_embedding` uses `[cos, sin]` not DDPM-conventional
`[sin, cos]`** — `model.py:240`. Functionally equivalent (the time MLP
absorbs the permutation), but inconsistent with the rest of the project's
diffusion modules and other paper reimpls. Cosmetic.

**N-6: Stage A lr=1e-4 at batch=16 is high under linear scaling** — paper
batch 64–128 implies lr ≈ 1e-4 was calibrated for a larger effective
batch. Blind impl batch 16 + lr 1e-4 is at the upper end. Recommend a
warmup schedule or lr=5e-5 for Stage A; flag as parameter to tune in the
first short run.

---

## Suggested fixes for Phase-2

1. `src/hfh_font/train.py:131` — add
   ```python
   init_from = train_cfg.get("init_from")
   if init_from:
       sd = torch.load(init_from, map_location="cpu")
       model.load_state_dict(sd, strict=False)
       print(f"[hfh_font] loaded init weights from {init_from}")
   ```
   before `optimizer = _build_optimizer(...)`.

2. `src/hfh_font/model.py:638` — add at top of `compute_sds_loss`:
   ```python
   assert diffusion.prediction_target == "x0", \
       "SDS placeholder only supports x0 prediction"
   ```
   and add a `TODO(phase-2)` comment pointing at the formal SDS gradient.

3. `src/hfh_font/model.py:614-620` — split the dropout knob into
   `cfg_dropout_refs`, `cfg_dropout_writer`, `cfg_dropout_script`,
   `cfg_dropout_char=0.0` and apply only in the last 10% of iters per
   paper §訓練配置.

4. `src/hfh_font/configs/model.yaml:20` — set `components_per_ref: 4` (the
   actually-used value) or `9` (the actually-allocated value for `8`);
   document the rounding behavior.

5. `tests/test_smoke.py` — add a third test
   `test_component_encoder_grad_flows_after_two_steps` that asserts
   `component_encoder.parameters()` accumulate gradient after two
   `compute_loss + step` iterations. Prevents future regressions where
   cross-attention gets accidentally severed.

---

## Suggested ablations (Phase-3, nice-to-have)

- **Pretrained-VAE vs random-init VAE** on Stage A reconstruction quality
  (PSNR / LPIPS on a 200-glyph held-out set). Quantifies weakness #1.
- **IDS-component tokens vs adaptive-pool tokens** on FID @ few-shot
  generation. Quantifies weakness #2 and the paper's "component-level
  reuse" claim.
- **CFG dropout p ∈ {0.0, 0.1 (ref-only), 0.1 (all-channels)}** — three-way
  comparison to verify the paper's narrow specification.
- **DDIM-10 vs trailing-DDPM-10** sampler ablation on FID. Quantifies
  weakness #4.

---

## Files inspected

- `paper_notes/02.md` (full)
- `reports/blind_impl.md` (full)
- `src/hfh_font/model.py` (full, all 742 lines)
- `src/hfh_font/train.py` (full)
- `src/hfh_font/dataset.py` (full)
- `src/hfh_font/sample.py` (full)
- `src/hfh_font/configs/{model,train_stage_{a,b,c}_*,data_stage_a}.yaml`
- `tests/test_smoke.py` (full)
- `shared/src/paper_reimpl_shared/diffusion/gaussian.py` (full)
- `shared/src/paper_reimpl_shared/data/legacy.py` (normalization grep only)

No contamination — every Phase-1 deviation from the paper is explicitly
tagged `[guessed]` or `[paper-cited]` in the worker's own decision log.
