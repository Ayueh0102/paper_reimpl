# DL Review — 07_moyun — Gate 1 (blind impl)

## Verdict: PASS-WITH-NITS

Blind status confirmed (`facts_code_url: null` per `paper_notes/07.md` §intro).
Reviewer did NOT peek at any external Moyun implementation. All judgments are
relative to (a) the paper notes shipped at
`paper_notes/07.md` + `reports/blind_impl.md`, and (b) standard SSM / DDPM /
DiT recipes (Gu & Dao 2024, Ho 2020, Peebles & Xie 2023, Zhu et al. 2024).
**Not CONTAMINATED.**

The implementation is faithful to the paper's three load-bearing claims:
1. pure-PyTorch S6 recurrence with **selective** Δ/B/C and a sequential scan
2. **bidirectional** scan via flip-add-divide-by-2
3. **TripleLabel = three distinct `nn.Embedding`s, summed**, with reserved
   `[NULL]` row at index 0 for CFG dropout
4. **adaLN-Zero** modulation (final linear of `cond_mlp` zero-init) with
   `LN(x) * (1 + scale) + shift` applied before SSM and FFN sublayers
5. VAE-absent design: `in_channels=4` for production latent, `in_channels=1`
   for smoke; the model is shape-agnostic and the train script forces 1 when
   `--synthetic` is set.

No FAIL-class defects found. Three nits and one math approximation are worth
flagging for Phase 2 github-diff.

---

## Checked

### S6 SSM correctness (pure-PyTorch)

- [x] **Selective Δ/B/C**: `x_to_BCdt` projects each token to
  `(dt_pre, B, C)` (`src/moyun/mamba_block.py:170-175`) — these are
  **functions of the input**, not learned constants. Matches S6 (Gu & Dao
  2024 §3.2 "selectivity").
- [x] **Δ = softplus(dt_proj(dt_pre))** strictly positive
  (`mamba_block.py:176`). `dt_proj.bias` is constant-initialized to `-5.0`
  (`mamba_block.py:145`) so `softplus(-5.0) ≈ 0.0067`, giving `A_bar = exp(Δ·A) ≈ 1`
  at init. Stable.
- [x] **A_log parameterization**: `A = -exp(A_log)` keeps eigenvalues
  negative real (`mamba_block.py:180`); init `-1..-d_state` per channel
  (`mamba_block.py:150`) gives HiPPO-LegT-flavored prior. Correct.
- [x] **A_bar = exp(Δ·A)** elementwise on diagonal A
  (`mamba_block.py:181`). Broadcasting shapes
  `(B,L,d_model,1) * (1,1,d_model,d_state) → (B,L,d_model,d_state)` are
  correct.
- [x] **B_bar = Δ·B** (Euler, `mamba_block.py:185`) — the paper notes
  acknowledge this is an approximation of ZOH; documented in module
  docstring (`mamba_block.py:22-27`). See **Nit 1** below.
- [x] **u_t = B_bar ⊙ x_conv** then **scan**
  (`mamba_block.py:188-197`): the recurrence
  `h_t = A_bar_t * h_{t-1} + u_t; y_t = (h_t · C_t).sum(-1)` is exactly the
  S6 update with diagonal A. Initial `h_0 = 0` correct.
- [x] **D residual + gate**: `y = y + D * x_conv; y = y * SiLU(z_gate)`
  (`mamba_block.py:200-203`). The expand=2 split (`in_proj` → `x_in, z_gate`)
  matches Mamba's gating idiom (`mamba_block.py:121, 163`).
- [x] **Gradient flow through scan**: the for-loop scan is fully
  differentiable PyTorch (no `.detach()`, no `torch.no_grad`). Verified by
  `tests/test_smoke.py:117-121` which checks
  `block.ssm.parameters()` all see non-zero grad.

### Bidirectional scan

- [x] Two **distinct** `SelectiveScanSSM` instances (`ssm` and `ssm_b`,
  `mamba_block.py:281-284`) — separate parameter sets, not weight-shared.
  Forward path: `y_f = ssm(h); y_b = ssm_b(h.flip(1)).flip(1); y = (y_f+y_b)*0.5`
  (`mamba_block.py:303-306`). Average matches Vision Mamba (Zhu et al.
  2024). See **Nit 2** for an alternative.
- [x] The `flip` is along the sequence axis (`dim=1`), correct for the
  `(B, L, d_model)` layout post-patchify.

### TripleLabel (three separate `nn.Embedding`)

- [x] **Three distinct modules** (`model.py:190-192`):
  ```python
  self.writer = nn.Embedding(writer_vocab + 1, hidden_dim)
  self.script = nn.Embedding(script_vocab + 1, hidden_dim)
  self.char   = nn.Embedding(char_vocab   + 1, hidden_dim)
  ```
  No weight-sharing, no shared backbone. Forward returns
  `e_writer + e_script + e_char` (`model.py:206-209`). Matches paper §3.4.
- [x] **`[NULL]` slot at index 0**, zero-initialized
  (`model.py:195-198`). `+1` slot in `Embedding` vocab size accommodates it
  (`model.py:190-192`). Real id range becomes `[1, vocab)`.
- [x] **`_resolve_id` shifts by +1** (`model.py:156-158`); `None` →
  zeros (route to `[NULL]`). `-1` (CFG drop sentinel) → `0` (`[NULL]`).
  Round-trip is consistent with `train._cfg_drop` (`train.py:54`:
  `out[drop_mask] = -1`).
- [x] **CFG dropout** in `compute_loss` (`train.py:86-98`): both `joint`
  and `independent` modes implemented; joint default matches Ho & Salimans
  2022. Drop mask applied to all three ids identically in joint mode.
- [x] **Distinct-modules invariant** enforced by
  `tests/test_smoke.py:203-219`
  (`test_triple_label_embeddings_are_distinct_modules`) — checks module
  identity AND that outputs on shared `ids=[1,2,3,4]` are different
  (random-init makes accidental allclose negligible).
- [x] **Each table receives its own gradient** verified by
  `tests/test_smoke.py:96-108` — the writer / script / char `.weight.grad`
  are each non-zero after the second backward (first backward is needed
  because cond_mlp[-1] is zero-init).

### adaLN-Zero modulation init

- [x] **`cond_mlp[-1]` final linear zero-init** (`model.py:298-301`):
  both weight and bias zeroed. Matches DiT §3.2 "adaLN-Zero".
- [x] **Modulation form `LN(x) * (1 + scale) + shift`**
  (`mamba_block.py:302, 312`). Note the `(1 + scale)` not `scale` — this
  ensures the block behaves as identity when `scale=0, shift=0` (i.e.
  zero-init state).
- [x] **Both SSM and FFN sublayers modulated** (`mamba_block.py:302-313`)
  with separate `(scale, shift)` pairs unpacked from a chunk-of-4
  `(scale_ssm, shift_ssm, scale_ffn, shift_ffn)` (`mamba_block.py:294`).
  Matches paper §3.4 "DiT-style scale-shift".
- [x] **LayerNorm with `elementwise_affine=False`**
  (`mamba_block.py:279-280`) — the affine is supplied by the modulation
  output, so disabling LN's own affine is correct (else double-affine).
- [x] **One matmul for all blocks**: `cond_mlp` projects to
  `4 * H * num_blocks` then `chunk(num_blocks, -1)` (`model.py:362`). Each
  block-chunk is `(B, 4H)` and `chunk(4, -1)` inside `VisionMambaBlock`
  splits to four `(B, H)` tensors. Correct.
- [x] **TripleLabel + time embedding combination**: `cond = e_total + t_emb`
  (`model.py:322`). Additive, consistent with DiT.

### VAE absence

- [x] **`in_channels=4` default in `MoyunConfig`** (`model.py:79`) — for
  the paper's VAE-latent mode. `in_channels=1` is the smoke path.
- [x] **`PatchEmbed` / `PatchUnembed` accept any `in_channels`**
  (`model.py:219-246`) via `Conv2d` / `ConvTranspose2d`. Shape-agnostic.
- [x] **Train script forces `in_channels=1` when `--synthetic`**
  (`train.py:210-211`):
  ```python
  if bool(getattr(args, "synthetic", False)):
      cfg.in_channels = 1
  ```
  Correct, since `SyntheticCalligraphyDataset` emits 1-channel grayscale.
- [x] **`model.yaml`** defaults to `in_channels: 4` (the production
  latent), with the smoke test overriding to 1. Matches blind-impl decision
  log A12.

### Diffusion / loss

- [x] **MSE eps-prediction** (`train.py:108`):
  `F.mse_loss(model_pred, diff_batch.target, reduction='mean')` — matches
  paper §3.5 DDPM default. `reduction='mean'` consistent with DDPM
  convention.
- [x] **`prediction_target='epsilon'`** in `train_stage_a_ttf.yaml:31`
  and propagated through `_build_diffusion` (`train.py:153`).
- [x] **β schedule linear 1e-4 → 2e-2 over 1000 steps**
  (`train_stage_a_ttf.yaml:27-30`) — DDPM defaults (`[guessed]` per
  blind_impl A15). Reasonable.
- [x] **Time embedding sin/cos + 2-layer SiLU MLP**
  (`model.py:135-147, 273-277`); dim = `hidden_dim` (`model.py:271-272`).
  Matches DiT.
- [x] **CFG sampler integration**: `sample.py:95` passes
  `cfg_uncond_drops_content=False` to the shared
  `GaussianDiffusion.sample`. The shared sampler's uncond branch passes all
  ids as `None`, which the model routes to `[NULL]`
  (shared/`gaussian.py:172-184`). Round-trip verified by
  `test_smoke.py:164-200`
  (`test_cfg_dropout_routes_to_null_embedding`): `writer=script=char=None`
  output is `allclose` (atol 1e-5) to passing `drop_ids = -1` (which becomes
  index 0 via shift). Clean.

### Training dynamics / hygiene

- [x] **Loss finite on smoke**: `test_smoke.py:78, 90, 130-131` —
  `torch.isfinite(loss)` and `torch.isfinite(p).all()` after one optim
  step.
- [x] **Seed set**: `_seed_everything` covers `random`, `numpy`, `torch`
  (+ cuda) in `train.py:118-123`. Called at start of `main`
  (`train.py:204`).
- [x] **`grad_clip=1.0`** in stage A yaml; applied via
  `torch.nn.utils.clip_grad_norm_` (`train.py:259`). Standard DDPM/DiT.
- [x] **AdamW lr=1e-4, β=(0.9, 0.999)** matches paper §3.5
  (`train.py:218-224`).
- [x] **No `.detach()` on critical path**: grep `mamba_block.py` and
  `model.py` — only `.detach().cpu()` in loss-log path (`train.py:109`),
  which is correct (numbers only, not used in backward).
- [x] **No silent `except:`** in src/moyun.

---

## Nice-to-have (PASS-WITH-NITS)

### Nit 1 — Euler vs ZOH discretization
`mamba_block.py:185` uses `B_bar ≈ Δ * B` (Euler) where the canonical
Mamba uses `B_bar = (A^{-1})(exp(Δ A) - I) B`, which for diagonal A
simplifies to `B_bar = ((exp(Δ A) - 1) / A) * B`. Euler underweights
contribution slightly when `Δ A` is non-small. At this paper's
`L=16` and small `Δ` from `softplus(-5)` init, the difference is
numerically tiny but worth flagging for Phase 2 — if the official repo
uses ZOH our outputs will diverge slightly on long runs.

Fix later (not blocking):
```python
# mamba_block.py:185 — replace Euler with ZOH
dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (B,L,d_model,d_state)
B_bar = (torch.expm1(dt_A) / A.unsqueeze(0).unsqueeze(0)) * B.unsqueeze(-2)
```

### Nit 2 — Bidirectional combine: `(y_f + y_b) * 0.5` vs concat-and-project
Vision Mamba (Zhu et al. 2024) actually concatenates `y_f` and `y_b` along
the channel axis and runs a Linear back to `d_model`; the blind impl uses
the simpler average. Mathematically the average can be re-derived as a
fixed concat+project, so this is not wrong, but if the official repo uses
the concat form, an ablation will show ours has half the bidirectional
mixing capacity. Flag for Phase 2 diff.

### Nit 3 — Sampler ignores `MoyunSampleConfig.seed`
`sample.py:78-84` builds `init_noise` from the seeded generator then
discards it (`_ = init_noise`) and relies on the shared sampler's internal
`randn`. The shared `GaussianDiffusion.sample` does not accept a
generator. So `MoyunSampleConfig.seed` is dead state for noise init.
Either:
- Drop `seed` from `MoyunSampleConfig`, or
- Plumb `init_image=init_noise` through the shared sampler signature.

Cosmetic for Gate 1; matters once we start reproducing eval grids.

### Nit 4 — `MoyunSampleConfig.in_channels` default mismatch with smoke
`sample.py:39` defaults `in_channels=4` (latent mode) while smoke uses 1.
Callers must override. Not a bug since `model.in_channels` is the source
of truth, but it's an easy footgun. Suggest reading `model.cfg.in_channels`
inside `sample` and asserting equality rather than re-specifying it.

### Nit 5 — Schedule scaling not documented per linear-rule
The blind_impl A17 acknowledges paper used global batch 768; we use 16
without scaling lr. DDPM is known to be robust here but Phase 3 might want
a sweep (`lr ∈ {2e-6, 1e-5, 5e-5, 1e-4}`). Already mentioned in A17 —
verbatim PASS.

---

## Suggested ablations (Phase 3)

- `bidirectional ∈ {True, False}` — confirms Zhu et al. 2024's "bi helps"
  claim on calligraphy patch sequences (L=16 is small enough that the
  asymmetry could be invisible).
- `label_dropout_mode ∈ {joint, independent}` — paper does not specify;
  independent dropout enables partial-condition generation (writer-only
  prompts, char-only prompts) which is a Moyun-relevant zero-shot mode.
- `prediction_target ∈ {epsilon, x0}` — `GaussianDiffusion` supports the
  toggle (shared/`gaussian.py:84-87`); paper does not pin this. (Note: the
  shared module does NOT support `v`-prediction; paper_notes hint at it
  but it's out of scope for this gate.)
- `d_state ∈ {8, 16, 32, 64}` — paper does not pin; community defaults
  range. Cheap on `L=16`.
- ZOH vs Euler `B_bar` (Nit 1) — once we have a longer training run.

---

## Files audited

- `/Users/Ayueh/Char/paper_reimpl/papers/07_moyun/src/moyun/mamba_block.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/07_moyun/src/moyun/model.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/07_moyun/src/moyun/train.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/07_moyun/src/moyun/sample.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/07_moyun/src/moyun/dataset.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/07_moyun/src/moyun/configs/model.yaml`
- `/Users/Ayueh/Char/paper_reimpl/papers/07_moyun/src/moyun/configs/train_stage_a_ttf.yaml`
- `/Users/Ayueh/Char/paper_reimpl/papers/07_moyun/tests/test_smoke.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/07_moyun/paper_notes/07.md`
- `/Users/Ayueh/Char/paper_reimpl/papers/07_moyun/reports/blind_impl.md`
- `/Users/Ayueh/Char/paper_reimpl/shared/src/paper_reimpl_shared/diffusion/gaussian.py`
  (sampler contract sanity-check only)
