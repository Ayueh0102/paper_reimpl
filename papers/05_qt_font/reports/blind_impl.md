# QT-Font Blind Implementation — Decision Log

Source-of-truth notes I consulted:

- `/Users/Ayueh/Documents/Obsidian Vault/research/papers/027_QT-Font四叉樹擴散字體_SIGGRAPH2024.md`
- `/Users/Ayueh/Char/paper_reimpl/reports/phase0_spec_table.md`, row **05_qt_font**

Tag convention (per AGENTS.md):

- `[paper-cited <section>]` — explicit statement / figure / table / quote in the note.
- `[guessed-<reason>]` — non-trivial choice not pinned by the note; reason in tag.

I have not read the official GitHub repo (`facts_code_url: null` per the
Phase 0 table — and the BLIND constraint would forbid it even if a URL were
known).

---

## Decision Log

### Architectural shape

1. **[paper-cited p.1-2]** Pipeline = `rasterised outline → sparse point cloud →
   quadtree → dual quadtree graph U-Net → discrete diffusion → output`. I keep
   all four conceptual stages: `QTFontModel` does the graph + diffusion; the
   pixel↔quadtree boundary is in `quantize_to_states` / `decode_states_to_image`.

2. **[paper-cited p.1]** Loss is "cross-entropy on quadtree node states". I
   interpret that as **D3PM x₀-prediction CE** with a uniform absorbing-state
   transition matrix `Q_t = (1-β_t) I + (β_t/K) 1·1ᵀ`. Justification: this is
   the simplest discrete-diffusion variant whose ELBO collapses to CE on x₀.

3. **[paper-cited p.1]** "dual quadtree graph network". I implement that as
   two parallel graph stacks (fine + coarse) with a pool between them. The note
   does not state whether messages flow simultaneously or sequentially; I picked
   *fine → pool → coarse → broadcast back* because it is the cheapest version
   that still exchanges information both ways.

4. **[paper-cited p.1]** "content-aware pooling reduces compute". Implemented
   as a 4-way softmax over a learned saliency MLP per parent — i.e. the parent
   feature is `Σ_k softmax(score_k) · child_feat_k`. The note does not give a
   formula, so this is the canonical attention-pool interpretation.

5. **[guessed-because-paper-vague]** Adjacency for the fine graph: 4-connectivity
   on the leaf grid (N/E/S/W). Quadtrees do not have an inherent sibling
   adjacency, but the paper's "graph network" claim requires *some* sibling
   edges. Pixel-adjacency over the leaf grid is the most defensible default.

6. **[guessed-because-paper-vague]** Adjacency for the coarse graph: 4-connectivity
   on the parent grid (`2^(d-1) × 2^(d-1)`). Same reasoning.

### Quadtree construction

7. **[guessed-major]** Use a **full saturated quadtree** of fixed depth, not the
   paper's adaptive sparse quadtree. The paper's pipeline is `outline → point
   cloud → adaptive quadtree`. Adaptive trees produce ragged per-sample
   structures, which require a graph library (PyG / DGL) or custom collate. To
   keep the Phase 1 sandbox dependency-light and the smoke test fast, I use a
   full tree (every sample has 4^depth leaves). This is the single largest
   deviation from the paper.

8. **[guessed-from-table]** Default depth = 4 → 16 × 16 leaf grid. Note 027
   only commits to "256/512 px output". Depth-4 keeps node count at 341
   total / 256 leaves, which is comparable to a 16-token ViT patch grid and
   fits comfortably in a 24 GB GPU.

9. **[guessed-because-paper-vague]** Default `n_states = 8` categorical bins per
   leaf. The note does not state K. Binary (K=2) loses sub-pixel intensity;
   8 bins ≈ 3-bit intensity is the smallest K that still represents anti-aliased
   stroke edges meaningfully.

10. **[guessed-because-paper-vague]** Quantisation = `adaptive_avg_pool2d` of the
    pixel image down to the leaf grid, then `(x+1)/2 · K` clamp-cast to long.
    The paper uses an outline-based construction; for synthetic / shared-smoke
    compatibility I quantise directly from the pixel tensor.

### Diffusion schedule

11. **[guessed-because-paper-vague]** Discrete diffusion `T = 100`. The note
    only mentions discrete diffusion qualitatively. D3PM image experiments
    in Austin et al. used T = 1000 for image-shaped categorical states but
    much smaller (~256) for token tasks. 100 keeps smoke wall-clock <10 s on
    CPU and is a reasonable starting point.

12. **[guessed]** β schedule = linear from 1e-4 to 0.02 (D3PM uniform default).
    The note does not specify.

13. **[paper-cited "訓練配置"]** AdamW β₁ 0.9, β₂ 0.999, weight decay 1e-2,
    lr 1e-4 cosine.

14. **[paper-cited "訓練配置"]** Pretrain effective batch 1024 via gradient
    accumulation; fine-tune batch 8. We default to `batch_size: 8` in YAML
    and leave grad-accumulation as a Phase 2 follow-up (current Phase 1 dry-run
    needs at most 1 step).

### Conditioning

15. **[guessed-because-paper-vague]** Conditioning injection = additive,
    broadcast over every leaf. The note does not specify FiLM, cross-attention,
    or AdaLN. Sum-then-broadcast is the simplest scheme that keeps every
    conditioning path receiving gradient (validated by the per-branch grad
    check in `test_smoke.py`).

16. **[guessed]** Per-id null tokens. For CFG dropout I add an extra "null" id
    at index `vocab_size` to each embedding. The note does not say whether
    QT-Font uses CFG, but the spec table (and most 2024 diffusion FFG papers)
    do; I keep `cfg_drop_prob: 0.1` as Stage A default.

17. **[guessed]** Style encoder = 3-conv CNN + adaptive-avg-pool + mean over
    `R` refs, masked by `ref_valid`. The note doesn't describe an explicit style
    encoder for QT-Font (most of the note is on quadtree + diffusion); for
    few-shot we need *something* that consumes the reference glyph stack, so I
    add this as the minimal lightweight encoder.

18. **[guessed]** Content encoder = 3-conv CNN with the same shape as the style
    encoder. The note doesn't specify a source-glyph encoder either, but the
    shared smoke harness provides a `content` tensor, and the paper's "few-shot
    font generation" task implies a content channel.

### Inference / sampler

19. **[guessed]** Reverse process uses `x₀-conditional` sampling
    (predict `x₀`, then re-noise to `t-1`). The full D3PM posterior
    `p(x_{t-1} | x_t, x₀) ∝ q(x_{t-1} | x₀) q(x_t | x_{t-1})` is exact but
    O(K²) per node per step; for K = 8 it's cheap, but the `x₀`-conditional
    form is what D3PM section 3.2 recommends in practice and is what we ship.

20. **[guessed]** Greedy argmax at the final step (`t=0`) instead of one more
    multinomial sample. Marginally cleaner sample grids; trivial to disable
    via the `greedy_final_step` kwarg.

### Pixel-space adapter for shared infra

21. **[guessed]** `QTFontModel.forward(x_t: pixel tensor, ...)` quantises `x_t`
    internally and decodes the predicted logits back to a pixel image. This
    keeps the model drop-in for the shared smoke harness and the shared
    `GaussianDiffusion` sampler, even though training uses the native discrete
    loss (`train.compute_loss`). The continuous gradient path through `x_t` is
    intentionally broken at the `.long()` quantisation — that's a feature of
    D3PM, not a bug.

### Things I did NOT implement / explicit follow-ups

22. **[guessed]** No actual point-cloud-from-outline extractor. Phase 2 (with
    the official code) is the right time to add a quadtree-from-Bézier route.

23. **[guessed]** No grad-accumulation orchestrator. The YAML notes the paper's
    1024-effective batch but the Phase 1 training loop runs a single
    optimiser step per batch.

24. **[guessed]** No manifest-backed dataset. Stage A/B/C YAMLs reference
    manifest filenames but the loader is currently `SyntheticDataset` only.

25. **[guessed-because-paper-vague]** No explicit content-aware-pooling
    regulariser term in the loss. The note hints at one, but no formula is
    given. The pooling module itself is differentiable, so any learning signal
    propagates through normal CE.

---

## Confidence summary

| Block | Confidence |
|---|---|
| D3PM uniform CE loss | high (only "discrete CE" choice consistent with the paper text) |
| AdamW / lr / weight decay | high (paper-cited) |
| Quadtree depth = 4 / n_states = 8 / T = 100 | low (all guessed) |
| Full vs adaptive quadtree | low (paper uses adaptive; we use full) |
| Conditioning injection style | low (additive vs FiLM unknown) |
| Style / content encoders | low (note silent) |
| Coarse-graph topology | medium (most natural choice) |

These confidence levels are what Phase 2 should pay attention to during the
github diff pass.
