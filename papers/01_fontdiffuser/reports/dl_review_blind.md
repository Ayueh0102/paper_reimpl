# DL Review — 01_fontdiffuser — Phase 1 (blind impl)

Reviewer: DL correctness only. Code-style / hardcoded-path / hygiene items are
out of scope (handled by the separate Code Reviewer pass).

## Verdict: PASS-WITH-NITS

Zero correctness FAILs found. The implementation matches standard DDPM
conventions, gradient flows through all three conditioning branches (content /
style / time), schedule is sane, and the smoke test covers forward + backward
+ 1 sample step. The author's six self-flagged weaknesses are all real but
none of them block Phase 2 advancement (Phase 2 = "open the official repo and
diff"). I add three new nits the author did not flag — most importantly a
train/sample CFG-uncond branch mismatch that will silently corrupt CFG > 1
sampling once Stage B launches.

## Checked

### Loss correctness
- [✓] `L_simple = F.mse_loss(ε̂, ε, reduction='mean')` matches Ho 2020 eq.14
  (simple objective) — `train.py:120`. Reduction `'mean'` is standard for
  DDPM; sum-vs-mean would only matter if combined with a custom lr scale, and
  the current lr (1e-4 → 2e-5 across stages) assumes mean.
- [✓] ε-prediction selected via `prediction_target='epsilon'` in all three
  stage configs (`configs/train_stage_{a,b,c}.yaml:31/24/22`). `predict_x0`
  inverts via `(x_t − √(1−ᾱ_t)·ε̂) / √ᾱ_t` — matches Ho eq.15 — see
  `shared/.../gaussian.py:129-132`.
- [✓] SCR formula is a sensible supervised NT-Xent — `train.py:47-71`. Math:
  `sim = z_pred @ z_true.T / τ`, `log_prob = sim − logsumexp_j sim`, masked
  by `labels[i]==labels[j]`, averaged over positives, mean over batch.
  Matches `paper_notes/01.md` eq.SCR up to the documented swap of
  "same-char-different-style" → "writer_id within batch".
- [✓] `z_true = scr_extractor(x_0)` under `torch.no_grad()` is correct
  (anchor branch, stop-grad). `z_pred = scr_extractor(x̂_0)` keeps the
  gradient path open so SCR backprops into the U-Net via predicted x₀ —
  `train.py:134-137`.
- [✓] Total loss combination `L = L_simple + λ_scr·L_scr` with λ_scr ramp
  0.0 → 0.1 → 0.2 across A/B/C is documented and consistent across the three
  train YAMLs.
- [✓] `scr_weight=0.0` ⇒ SCR branch skipped entirely (`train.py:125`); no
  wasted forward pass and no risk of dividing by zero in early Stage A.

### Gradient flow
- [✓] No accidental `.detach()` or `torch.no_grad()` on the critical path.
  Only intentional uses: (a) `z_true = no_grad(extractor(x0))` for SCR
  anchors, (b) `loss.detach().cpu()` for logging — both correct.
- [✓] Smoke test (`tests/test_smoke.py:71-81`) explicitly asserts non-zero
  gradient on `content_encoder`, `style_encoder`, and `unet` after one
  backward — catches the canonical "conditioning path disconnected" bug.
- [✓] Second smoke test (`tests/test_smoke.py:111-156`) verifies the SCR
  path does not disconnect the U-Net (unet_grad > 0 with scr_weight=0.1).
- [✓] Style null-token (`model.py:544`) gets gradient when `ref_valid=False`
  via `torch.where` — invalidated samples route through the learned null
  rather than zero, so the cross-attention block is always defined.
- [✓] Time embedding path is always live (`model.py:593-594`).

### Schedule & sampler
- [✓] β linear schedule, `β_1=1e-4`, `β_T=2e-2`, `T=1000`. With T=1000 →
  `ᾱ_T = ∏(1−β_t) ≈ 4e-5` (standard Ho 2020 value, effectively zero), so
  forward process reaches near-pure noise — `gaussian.py:96-105`.
- [✓] β monotonically increasing by construction of `torch.linspace`.
- [✓] Sampler matches training: `p_sample` and `p_sample_ddim` both call
  `_model_pred` and then `predict_x0` with the same `prediction_target`
  (`gaussian.py:273, 321`). No ε-vs-x₀ convention mixing.
- [✓] Time embedding: sin/cos with `dim=base_channels=64`, followed by
  `Linear(64→256) → SiLU → Linear(256→256)`. Standard Ho 2020 / Improved-DDPM
  recipe — `model.py:108-119, 378-382`.
- [✓] DDIM sampler reconstructs `pred_noise` from the predicted x₀ via the
  schedule — `gaussian.py:326-327`. Consistent with the ε-prediction
  training target.
- [✓] `posterior_variance`, `posterior_mean_coef1/2` precomputed correctly
  (`gaussian.py:108-110`); matches Ho eq.6-7.

### Conditioning paths
- [✓] Style encoder output reaches the U-Net's noise prediction via RSI
  cross-attention. `RSIBlock` is invoked at every `attn_resolutions` stage
  in both the down path (`model.py:483`) and up path (`model.py:508`), plus
  the middle block (`model.py:491`). With default `attn_resolutions=(16,)`,
  RSI fires at the bottleneck only (3 sites total: down-16, mid-16, up-16).
- [✓] Content encoder output reaches the U-Net via MCA fusers at **every**
  stage (`down_mca`, `up_mca` lists, fired unconditionally in
  `model.py:480, 505`). MCA gate is zero-init so the path starts as identity
  + learns — author's own design choice, documented at `model.py:343-345`.
- [✓] MCA "gated-add" substitute for the unspecified paper operator
  (`MCAFuse.forward`, `model.py:347-354`): `h ← h + sigmoid(W_g · proj(c)) · proj(c)`.
  Reasonable choice — at minimum it preserves the identity at init and
  allows gradient to flow into the content encoder from step 0.
- [✓] CFG dropout `p=0.1` implemented at `train.py:105-110`. Drops **only
  the style reference** (`ref_valid[drop, :] = False`), not the content. Per
  paper §3, content is the source-glyph identity and must always be
  available — this matches the author's intent and `paper_notes/01.md:170-172`.
- [✓] When ref is dropped, the style branch falls through to the learned
  `style_null_token` (`model.py:567-570`), so the cross-attention block is
  always well-defined. Differentiable through `torch.where`.

### Data normalization
- [✓] All data lands in [-1, 1]:
  - real glyphs: `load_grayscale_tensor` returns `array * 2.0 - 1.0`
    (`shared/.../data/legacy.py:65`)
  - `load_content_tensor` clamps to [-1, 1]
    (`shared/.../data/legacy.py:104`)
  - synthetic dataset uses `torch.linspace(-1, 1, ...)` + clamp
    (`shared/.../data/legacy.py:264-273`)
- [✓] Predicted x₀ in `GaussianDiffusion.predict_x0` is clamped to [-1, 1]
  (`gaussian.py:128, 132`), matching the data range that the SCR extractor
  was implicitly trained on.

### Three-stage curriculum coherence
- [✓] Stage A: synthetic TTF proxy, `lr=1e-4`, 5k steps, `scr_weight=0`,
  `cfg_drop=0.1`. Clean denoiser warm-up — SCR off as documented (paper
  expects the style extractor to be pretrained between Stages A and B).
- [✓] Stage B: manifest-backed multi-writer, `lr=5e-5`, 20k steps,
  `scr_weight=0.1`. SCR turns on, lr halves. Coherent.
- [✓] Stage C: target style fine-tune, `lr=2e-5`, 50k steps,
  `scr_weight=0.2`, `batch_size=8` (halved — likely for higher-res inputs
  later). Coherent.
- [✓] All three stages use the same β schedule + T + prediction_target —
  consistent. Switching schedule between stages would break the
  Stage-A checkpoint hand-off.

### Author's self-flagged weaknesses (assessed)

| # | Item | Verdict |
|---|------|---------|
| 1 | Output conv NOT zero-init | **PASS** — author's rationale (gradient must reach content/style branch on iter 0 for smoke test) is sound. Real cost is one extra epoch of warm-up. Trade-off, not a correctness defect. |
| 2 | No EMA on diffusion model | **PASS-WITH-NIT** — fine for Phase 1 smoke. Must be added before Stage B handoff; recommend decay=0.9999, applied via shared `paper_reimpl_shared.train.ema` (build a wrapper if shared lacks one). |
| 3 | SCR negatives = within-batch writer_id, not "same-char-different-style" | **PASS** — author flagged + documented. Paper's negative-mining needs a structured batch sampler that is not yet in scope. Within-batch supervised contrastive is a strict generalization (more negatives per anchor) and well-defined. Suggest as a Phase 2 ablation. |
| 4 | MCA fuser = gated-add (invented) | **PASS** — paper does not specify the operator. Gated-add with zero-init gate is conservative and matches Phase 2 review prep (decision is listed in `blind_impl.md:152-156` as a target diff). |
| 5 | RSI placement = attn_resolutions only | **PASS** — `attn_resolutions=(16,)` means RSI runs at down-16, middle, up-16. Paper note literally says "RSI cross-attn on reference"; placement is unspecified. The Improved-DDPM convention is bottleneck-only and matches FontDiffuser's "high-level structural deformation" framing. Phase 2 diff target. |
| 6 | StyleExtractor untrained (cold-frozen) | **PASS-WITH-NIT** — fine for Phase 1 smoke (SCR weight is 0 in Stage A). Will produce noisy SCR loss in Stages B/C: cold-init means embeddings are random projections and the contrastive signal is mostly noise. Must pretrain the extractor on writer-id classification **before** Stage B launches; otherwise SCR is doing nothing useful and worse may inject noise into U-Net grads. |

## Required fixes (block Phase 2 advancement)

None. Verdict is PASS-WITH-NITS, not FAIL.

## Nice-to-have (don't block; address before Stage B real-data launch)

1. **Train/sample CFG uncond-branch mismatch** — `train.py:105-110` drops
   only `ref_valid`, but `gaussian.py:165-178` constructs the uncond branch
   by **also zeroing the content image**. The (zero-content, no-ref)
   combination is therefore never seen during training; CFG scaling at
   sample time pulls toward an out-of-distribution prediction. Two ways to
   fix:
     - **Match train to sample**: when `cfg_drop_prob` fires, also zero
       `content` (cleanest; aligns with the shared sampler's existing
       behavior). Tradeoff: forces the model to learn a "content-free"
       mode, which costs capacity.
     - **Match sample to train**: in `gaussian._model_pred`, the uncond
       branch should keep `content` and only zero `ref_valid` (a
       FontDiffuser-specific override of the shared utility — could be a
       `cfg_uncond_zero_content: false` flag plumbed into
       `GaussianDiffusion`).
   The second is closer to the paper's intent ("classifier-free guidance
   over style reference"). Either fix is acceptable as long as train and
   sample agree. Recommend keep content, drop only ref.

2. **`predict_x0` clamping inside SCR backward** — `gaussian.py:128/132`
   clamps predicted x₀ to [-1, 1]. When the model is early in training and
   the predicted x̂_0 falls outside [-1, 1] (extremely common at high t),
   the clamp kills gradient on those samples. SCR loss therefore gets a
   biased gradient signal (only "in-range" predictions contribute). Two
   options: (a) drop the clamp when path is used inside the SCR forward
   (clone with `clamp` only at sample time), or (b) replace `clamp` with a
   soft clip (`torch.tanh`). Low priority for blind smoke; revisit when SCR
   actually activates.

3. **SCR self-pair in numerator** — `train.py:67-70` builds `label_eq` from
   `labels.unsqueeze(1) == labels.unsqueeze(0)`, which is True on the
   diagonal. Standard SupCon (Khosla 2020) excludes the self-pair from the
   positive set. Mathematically, since `z_pred[i]` and `z_true[i]` are
   different embeddings (predicted vs anchor), the diagonal is **not** the
   trivial self-pair — it is "same sample, different views" which is
   exactly what SupCon counts as a positive. So this is **probably fine** in
   the asymmetric (pred vs true) formulation. Just worth a sanity comment
   in code so a future reader does not think it's a bug.

4. **`style_null_token` shape inconsistency** — `model.py:559-561`: when
   `ref_images is None`, the null token is broadcast as `[B, 1, embed_dim]`
   (L=1), but when ref is present-and-masked the null fills `[B, L,
   embed_dim]` (L=256 at 128px input). The attention block handles both,
   but downstream code that aggregates over `L` (none currently) would see
   a different distribution. Low-priority polish.

5. **EMA hook before Stage B** — author #2 above. Wire it now (1-day task)
   so the Stage B checkpoint is the EMA model, not the raw weights.

## Suggested ablations (optional, Phase 2/3)

- **MCA fuser**: gated-add (current) vs concat-then-1×1 vs no MCA (ablate
  decision 4 in `blind_impl.md`). Measure stroke-completeness on hard
  characters.
- **RSI placement**: bottleneck-only (current) vs every-attn-resolution
  vs every-resolution. Measure style-transfer fidelity.
- **CFG drop policy**: drop ref only (current) vs drop both ref+content
  (full-uncond). Pairs with nit #1 above.
- **β schedule**: linear (current) vs cosine. Improved-DDPM showed cosine
  helps at high resolutions; FontDiffuser is at 128/256px so it could
  matter.
- **λ_scr ramp**: 0/0.1/0.2 (current) vs 0/0.05/0.05 (gentler). Author's
  ramp is aggressive; a gentler ramp may stabilize once SCR extractor is
  cold.
- **Style extractor backbone**: shallow CNN (current) vs pretrained VGG-19
  vs ResNet-50. Standard style-transfer convention is a pretrained backbone
  for the style space; the paper does not say, but it would be a meaningful
  diff.

## Contamination check

No evidence of GitHub peeking. The author's `blind_impl.md` section "Things I
would peek at GitHub for (if allowed)" (lines 149-164) is consistent with the
gaps observed in the code (channel widths guessed, MCA fuser invented, RSI
placement guessed). Verification statement at `blind_impl.md:20-23` is
explicit: no third_party clone, no FontDiffuser-main reference, no WebFetch.
The code's open questions (paper_notes/01.md:216-228) are exactly the same
list — consistent diary across artifacts.

Verdict for contamination: **CLEAN**.
