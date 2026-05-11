# DL Review — 05_qt_font — Gate 1 (Blind Reimpl)

## Verdict: PASS-WITH-NITS

Contamination check: **CLEAN**. Both `paper_notes/05.md` (L4-8) and
`reports/blind_impl.md` (L13-16) explicitly state the official repo has not
been consulted (`facts_code_url: null`). No `# from official repo`, no
copied identifier names, no suspicious constants. Decision log faithfully
tags every choice as `[paper-cited]` or `[guessed-...]`.

The implementation is internally consistent and mathematically correct for the
choices it commits to. The single largest deviation from the paper (full
saturated quadtree vs adaptive sparse) is **explicitly documented** as a known
gap (decision #7, blind_impl.md L52-60), not a hidden bug — so it does not
block Gate 1. Discrete diffusion Q_t / Q̄_t closed form, gradient flow through
the content-aware pool, conditioning paths, and pixel-decoder shape contracts
all check out. Training loss is finite, smoke tests assert per-branch gradient.

---

## Checked

### Loss correctness

- [x] **D3PM uniform Q_t formula matches Austin et al. 2021**
  `src/qt_font/model.py:230` — `alpha_bar * one_hot + (1 - alpha_bar) / K`
  is the correct closed form for the cumulative `Q̄_t = Π Q_s` under
  `Q_t = (1-β_t)I + (β_t/K)·11ᵀ`. Identity and `(1/K)·11ᵀ` commute under
  matmul, so the cumulative collapses to `ᾱ_t · I + (1-ᾱ_t)/K · 11ᵀ` where
  `ᾱ_t = Π(1-β_s)`. Matches the formula stated in
  `paper_notes/05.md` §3 L77-78.
- [x] **CE loss is x_0 prediction CE on per-leaf logits**
  `src/qt_font/model.py:243-249` (`loss_x0_ce`) and
  `src/qt_font/train.py:103` — `F.cross_entropy(logits, x0_states)` with
  default `reduction='mean'`. This is the D3PM x_0-parameterisation auxiliary
  loss the note describes as "cross-entropy on quadtree node states".
- [x] **`reduction='mean'`** (PyTorch CE default) — consistent for lr scaling.
- [x] **β schedule linear 1e-4 → 0.02** `src/qt_font/model.py:208` and yaml
  `configs/train_stage_a_ttf.yaml:24-25`. Matches D3PM-image default.
- [x] **q_sample produces valid indices in [0, K-1]** asserted by
  `tests/test_smoke.py:75-83` (`test_d3pm_q_sample_valid_indices`).

### Gradient flow

- [x] **No accidental detach on critical path.** `grep '.detach()' src/`
  returns nothing. Only `torch.no_grad()` usages are
  `sample.py:21,84` (inference) and `train.py:105` (accuracy metric only).
- [x] **Per-branch gradient asserted by smoke test**
  `tests/test_smoke.py:113-129` explicitly checks that
  `state_embed / content_encoder / style_encoder / char_embed / writer_embed /
  fine_layers / coarse_layers / pool / head` all receive non-zero gradient.
  This covers the ContentAwarePool gradient-flow concern from the rubric.
- [x] **ContentAwarePool is fully differentiable** `src/qt_font/model.py:300-326` —
  softmax over learned scores, weighted sum of child features. `masked_fill`
  with `-inf` only on invalid (negative) child slots, which never occur at
  the penultimate level of a full tree, so no NaN risk in practice.
- [x] **Conditioning paths reach loss**: time / content / style / char_id /
  writer_id / script_id are all summed into `cond` (model.py:534-570) and
  broadcast to every leaf via `cond.unsqueeze(1)` at both fine
  (model.py:588) and coarse (model.py:598) stacks. Smoke test confirms
  non-zero gradient on each branch.
- [x] **CFG dropout implemented** `src/qt_font/train.py:38-52, 82-90` —
  stochastic replacement of char/writer/script ids with a learned null id
  (the +1 row in each embedding, model.py:489-494).
- [x] **grad_clip = 1.0** `src/qt_font/train.py:235` and yaml.

### Quadtree topology / batching

- [x] **Full saturated quadtree, depth=4, 256 leaves on 16×16 grid**
  `src/qt_font/model.py:39-75` builds `parent_of` / `child_of` index tensors.
  `tests/test_smoke.py:51-66` verifies determinism and node count
  `(4^(d+1)-1)/3 = 85` at depth=3.
- [x] **Batchable: every sample has the same node count** by construction.
  This is the deliberate trade vs adaptive sparse trees.
- [x] **Leaf 4-connectivity adjacency on the 2^depth grid**
  `src/qt_font/model.py:78-102`. Coarse-graph adjacency reuses the same
  helper at depth-1 (model.py:469). Both are documented guesses but
  reasonable defaults.
- [⚠] **NIT — known deviation, not a FAIL**: paper specifies an *adaptive*
  sparse quadtree from rasterised outline; we use a full saturated tree
  quantised from `adaptive_avg_pool2d`. Documented at
  `reports/blind_impl.md:52-60` (decision #7) with explicit follow-up
  marker. Acceptable for Gate 1; flagged for Phase 2 diff against the
  official repo.

### Pixel decoder shape consistency

- [x] **`quantize_to_states` enforces `H % 2^depth == 0`**
  `src/qt_font/model.py:123-124`. At default depth=4, image_size=128 →
  128%16=0; at smoke depth=3, image_size=32 → 32%8=0.
- [x] **`decode_states_to_image` round-trips to (B,1,H,W)**
  `src/qt_font/model.py:133-158`. Bilinear upsample only when image_size
  ≠ grid; identity passthrough otherwise. Output is
  `E[bin_center]` in [-1,1] so it stays in the data range.
- [x] **forward() pixel-in / pixel-out adapter** `src/qt_font/model.py:611-652`
  composes quantize → graph denoise → decode. Smoke test
  `test_pixel_adapter_forward` asserts shape `(2,1,32,32)` and finiteness.
- [x] **Training loss path bypasses the decoder** — `train.compute_loss`
  calls `predict_logits_from_states` (model.py:680) on integer states
  directly, so the discrete CE works on raw logits, not pixel-roundtripped
  ones. Correct.
- [x] **Sampler `sample_image`** `src/qt_font/sample.py:84-109` decodes
  sampled state indices through `decode_states_to_image`. Smoke test
  `test_sample_image_shape` asserts shape + range.

### Conditioning paths

- [x] **time → sinusoidal + 2-layer MLP**
  `src/qt_font/model.py:257-267, 480-484, 535-537`.
- [x] **content image → CNN → linear proj → cond**
  `src/qt_font/model.py:362-383, 485-486, 539-540`.
- [x] **reference glyphs → CNN + mean over R, masked by ref_valid**
  `src/qt_font/model.py:329-359, 542-544`. `ref_valid` mask correctly
  prevents padded slots from contributing.
- [x] **char_id / writer_id / script_id → embedding with null id at
  `vocab_size`** `src/qt_font/model.py:489-494, 546-569`. CFG-compatible.
- [x] **Conditioning is broadcast to every leaf via `cond.unsqueeze(1)`**
  model.py:588, and also added to coarse features at model.py:598. Both
  the fine and coarse stacks see the conditioning.

### Training dynamics

- [x] **Loss is finite on smoke (B=2, depth=3, K=4, T=10)**
  `tests/test_smoke.py:107` asserts `torch.isfinite(loss)`.
- [x] **AdamW β=(0.9, 0.999), wd=0.01, lr=1e-4** `src/qt_font/train.py:210-215`
  and yaml — matches paper-cited values in `paper_notes/05.md` §6.
- [x] **Seed set for `random / numpy / torch / cuda`**
  `src/qt_font/train.py:122-127`.

### Sampler

- [x] **Discrete reverse process: x_0 parameterisation, re-q-sample to t-1**
  `src/qt_font/sample.py:56-81`. Matches the algorithm in
  `paper_notes/05.md` §8 L188-195.
- [x] **Greedy argmax at final step** controlled by `greedy_final_step`
  kwarg (`sample.py:71-72`); default True.

---

## FAILs

**None.** No blocking issues for Gate 1.

---

## Nice-to-have (PASS-WITH-NITS backlog)

1. **`src/qt_font/model.py:631`** — docstring references
   `:func:`compute_native_loss`` but the actual symbol is
   `qt_font.train.compute_loss`. Rename in docstring.
2. **`src/qt_font/model.py:115-117`** — `quantize_to_states` averages content
   across channels via `pooled.mean(dim=1)`. This is fine for the typical
   1-channel grayscale target but if Stage B/C target image ever becomes
   multi-channel (it currently is not — target is always 1-ch glyph,
   content is multi-ch) this would silently average them. Add an assert
   `C == 1` to be safe.
3. **`src/qt_font/model.py:208`** — β schedule is built on `device` at
   `D3PMUniform.__init__`, but `q_probs` (model.py:227) calls
   `self.alphas_cumprod.to(x0.device)` every step. Inconsistent — either
   pin the buffer on the correct device at init or always lazy-transfer.
   No correctness impact, minor allocation churn.
4. **`src/qt_font/sample.py:103-106`** — building `pseudo_logits = (one_hot * 10) - 5`
   and then re-softmaxing inside `decode_states_to_image` to compute
   `E[bin_center]` is a roundabout way of getting the argmax bin centre.
   Could index `bin_centers[states]` directly. Cosmetic.
5. **`src/qt_font/model.py:228`** — `alpha_bar.view(-1, 1, 1)` assumes
   leaf-state shape `(B, L, K)`. Works because `K` broadcasts on dim 2,
   but a comment clarifying the broadcast intent would help future readers.
6. **`tests/test_smoke.py`** — smoke test confirms gradient on `pool` but
   does **not** assert gradient on `parent_to_child` (the coarse→fine
   broadcast projection, model.py:506). Add to the branches dict for full
   gradient-flow coverage.
7. **D3PM training convention**: `q_sample` is called with `t` drawn from
   `[0, T-1]` (model.py:240-241), so `t=0` means "one step of noise added",
   not "clean x_0". Internally consistent with the sampler
   (`for t in reversed(range(T))`), but worth a 1-line comment in
   `D3PMUniform.__init__` so reviewers don't mistake it for an off-by-one.

---

## Suggested ablations (optional)

- **Full vs adaptive quadtree** — once Phase 2 unblocks reading the official
  repo, run an A/B at depth=4, 128 px: ours (full) vs ragged adaptive on the
  same TTF-pretrain subset. Hypothesis from blind_impl.md decision #7:
  expressivity should be similar; speed/memory differs.
- **n_states ablation**: K ∈ {2, 4, 8, 16} on Stage A. The paper does not
  pin K; n_states=8 is a guess (decision #9). Want to know if 4 bins lose
  edge anti-aliasing visibly or if 16 helps.
- **Conditioning injection**: additive (current) vs FiLM vs AdaLN on the
  per-leaf features. blind_impl.md decision #15 flags this as low
  confidence.
- **T ablation**: T ∈ {50, 100, 200} discrete steps. T=100 is guessed
  (decision #11).

---

## Files reviewed

- `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/model.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/train.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/sample.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/dataset.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/__init__.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/tests/test_smoke.py`
- `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/configs/model.yaml`
- `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/configs/train_stage_a_ttf.yaml`
- `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/paper_notes/05.md`
- `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/reports/blind_impl.md`
