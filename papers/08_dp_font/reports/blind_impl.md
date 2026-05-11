# Blind Implementation — DP-Font (08)

Decision log for the Phase 1 blind reimplementation. Every non-trivial
choice is tagged `[paper-cited <source>]` when sourced from the Obsidian
paper note or the Phase 0 spec table, or `[guessed-because-paper-vague]`
when we filled in a gap. Tag `[blind-impl-convention]` records tooling
decisions that are paper-neutral.

## Source material consulted

1. `/Users/Ayueh/Documents/Obsidian Vault/research/papers/023_DP-Font書法擴散
   PINN_IJCAI2024.md`
2. `/Users/Ayueh/Documents/Obsidian Vault/research/papers/DP-Font_Dual_
   Path_Font_Generation.md`
3. `~/Char/paper_reimpl/reports/phase0_spec_table.md`, row 08.
4. `~/Char/paper_reimpl/AGENTS.md`.
5. `~/Char/paper_reimpl/shared/src/paper_reimpl_shared/...`.
6. `~/Char/paper_reimpl/docs/REVIEW_RUBRIC.md`.

Neither `~/Char/paper_reimpl/third_party/` nor any GitHub search / fetch
was issued. The paper has `facts_code_url: null` in both Obsidian
sources, so no official repo could have been consulted even if allowed.

## Decisions

### Architecture

1. **DDPM U-Net backbone** `[paper-cited 023_DP-Font §"訓練配置"]`. Note
   states "DDPM (UNet 主幹) + PINN 物理損失". The shared
   `GaussianDiffusion` handles the forward / reverse process.

2. **80×80 input resolution** `[paper-cited 023_DP-Font §"訓練配置"]`.
   Note explicitly says "圖像大小 80×80". Our model.yaml defaults
   `image_size: 80`. The smoke test uses 32 px so it can fit in CPU memory
   in a second.

3. **Channel mult `(1, 2, 2, 4)`** `[guessed-because-paper-vague]`. Paper
   does not state widths. The chosen mult gives 4 stages at 80 -> 40 ->
   20 -> 10 px, putting attention at 10 px (matches DDPM's common 16-or-
   smaller-attn rule of thumb scaled to the 80 px input).

4. **`base_channels = 64`** `[guessed-because-paper-vague]`. DDPM 64-px
   default. Light enough to fit at 80 px on a single 3090 (paper's stated
   hardware) at batch 16.

5. **Self-attention only at `10×10`** `[guessed-because-paper-vague]`.
   Paper does not say. 10 px is the smallest spatial resolution available
   given the 80-px input and 4 stages; self-attention there matches the
   "attention at bottleneck" pattern of Improved-DDPM.

6. **No reference-image path** `[paper-cited row 08]`. Phase 0 row says
   "conditioning = Multi-attribute guidance (writer ID, ink intensity,
   font size, etc.) + stroke order". DP-Font is NOT few-shot; the model
   accepts `ref_images` / `ref_valid` only for shared-API compatibility
   and explicitly ignores them.

### Multi-attribute guidance

7. **Writer / script / char as categorical embeddings, plus a learnable
   null slot** `[paper-cited row 08 + blind-impl-convention]`. Each
   embedding table has `vocab_size + 1` entries; the trailing id is the
   "null" used by classifier-free guidance dropout. This mirrors Stable-
   Diffusion's learnable null context vector.

8. **Stroke-order embedded as token sequence + positional embedding,
   pooled by masked mean** `[guessed-because-paper-vague]`. Paper says
   stroke order is a "fine-grained constraint" but does not specify the
   encoder. Mean-pooled token sequence keeps the conditioning vector at
   constant width across all stroke-order lengths, which is the simplest
   plumbing for FiLM-style modulation. A more aggressive alternative (run
   the stroke sequence through a small Transformer and cross-attend into
   the U-Net) is left for Phase 2.

9. **Stroke vocab size 36** `[guessed-from-public-DB]`. The cjklib /
   Make-Me-a-Hanzi inventory has ~32 atomic stroke types plus compound
   forms; 36 covers both with slack. Phase 2 should align this with
   whichever DB we plumb.

10. **Stroke-order sequence length 32** `[guessed-because-paper-vague]`.
    Covers the bulk of Liu Gongquan / Yan Zhenqing characters; longer
    sequences are truncated.

11. **Ink intensity & font size as continuous scalars in `[0, 1]`**
    `[paper-cited row 08 + guessed-because-paper-vague]`. Phase 0 row 08
    lists "ink intensity, font size" but does not pin their range or
    encoding. A 2-element scalar MLP is the minimal viable hook.

### PINN losses (the largest guesses — paper publishes no PDE form)

12. **Three sub-terms: ink diffusion, nib motion, stroke continuity**
    `[paper-cited 023_DP-Font §"核心方法論"]` + `[guessed-because-paper-
    vague]` for the exact PDE form. Paper note says PINN models "毛筆運動 +
    墨擴散物理方程", but the two notes give no equations or coefficients.
    Our three differentiable surrogates are:
      * `L_diffusion`: steady-state isotropic diffusion `ν ∇²I + s = 0`
        with the background mask zeroing the source term.
      * `L_nib`: L1 norm of the Laplacian of the skeleton-aligned channel,
        masked by the predicted ink region.
      * `L_continuity`: speckle penalty (3x3 neighbour-mean check inside
        the ink region).

13. **Per-term weights `weight_diffusion = weight_nib = 1.0,
    weight_continuity = 0.5`** `[guessed-because-paper-vague]`. Continuity
    is the noisiest term at init (random networks produce a lot of
    speckle), so we down-weight it; tunable per stage.

14. **Master weight `λ_PINN`: 0.0 (Stage A) -> 0.05 (Stage B) -> 0.1
    (Stage C)** `[guessed-because-paper-vague]`. Paper says PINN is
    jointly optimised but does not give a weight; ramping from 0 makes
    Stage A a clean denoiser warm-up so the PINN term sees a sensible x0
    prediction once it activates.

15. **PINN evaluated on predicted x0, not on `eps`** `[guessed-because-
    paper-vague]`. The physical priors are over the ink density I, which
    corresponds to x0 (not the noise). For epsilon-prediction backbones we
    invert via the closed-form DDPM relation already implemented in
    `paper_reimpl_shared.diffusion.gaussian.GaussianDiffusion.predict_x0`.

16. **Soft masks (sigmoid) for ink/background separation** `[blind-impl-
    convention]`. Hard thresholds would block gradient at the mask
    boundary, defeating the "PINN must be differentiable" requirement.

### Diffusion schedule

17. **Cosine β schedule, T = 1000** `[guessed-because-paper-vague +
    extension]`. Paper note says "DDPM (UNet 主幹)" and does not pin T.
    Cosine is the Improved-DDPM default and tends to be better at small
    resolutions like 80 px than linear.

18. **Epsilon prediction** `[guessed-because-paper-vague]`. DDPM 2020
    default; the shared sampler also supports x0-prediction so we can
    flip via the train yaml.

19. **Linear β endpoints `1e-4, 2e-2`** (used only when `cosine` is
    overridden in the yaml) `[blind-impl-convention]`. DDPM standard.

### Training loop

20. **AdamW (β=(0.9, 0.999), weight_decay = 0)** `[blind-impl-
    convention]`. DDPM standard.

21. **`learning_rate` 1e-4 -> 5e-5 -> 2e-5** across Stage A / B / C
    `[guessed-because-paper-vague]`. Standard anneal-as-you-fine-tune
    recipe.

22. **Gradient clip 1.0** `[blind-impl-convention]`. Cheap insurance, no
    paper anchor.

23. **CFG dropout `p = 0.1`, applied per attribute independently**
    `[paper-cited row 08]` + `[blind-impl-convention]`. Phase 0 row 08
    lists CFG with `ω ∈ [0, 1]`; we choose Ho & Salimans 2022 default
    `p = 0.1` for the per-attribute Bernoulli.

24. **Single optimiser, single forward / backward per step** `[blind-
    impl-convention]`. PINN does not need a separate optimiser — both
    L_simple and L_PINN are differentiable against the same model
    parameters.

### Data plumbing

25. **Stroke order synthesised deterministically from char + writer hash**
    `[guessed-because-paper-vague + extension]`. Paper assumes real
    stroke-order labels; we have none. The hash makes the synthetic
    sequence deterministic so a smoke run is reproducible. Stage B/C must
    swap in a real lookup before launching real training.

26. **Ink intensity & font size synthesised from char + writer hash**
    `[guessed-because-paper-vague + extension]`. Same rationale.

27. **Content channels at Stage A = `[bitmap]`, Stage B/C = `[bitmap,
    skeleton]`** `[blind-impl-convention + paper-cited]`. The PINN nib-
    motion term reads the skeleton channel when available; we plumb it
    through `skeleton_channel_index` in the train yaml. Stage A keeps
    bitmap-only so the PINN term remains off until the model is
    warm-started.

### Sampler

28. **DDPM and DDIM both supported via the shared sampler**
    `[paper-cited row 08]` (DDPM) + `[blind-impl-convention]` (DDIM
    offered as a faster qualitative inspection sampler).

29. **`cfg_uncond_drops_content = False` at sample time** `[blind-impl-
    convention]`. DP-Font's training drops categorical attributes, not
    content. Without this flag the shared sampler would zero the content
    tensor in the uncond branch, pulling toward an out-of-distribution
    prediction never seen during training. (Same lesson the FontDiffuser
    reimpl learnt — fix applied here from day 1.)

30. **Frozen-condition adapter for stroke / scalar fields at sample
    time** `[blind-impl-convention]`. The shared `GaussianDiffusion.sample`
    method does not natively forward DP-Font's extra kwargs; we wrap the
    model in a tiny module that captures `stroke_order`, `ink_intensity`,
    `font_size` once on construction and re-injects them on every
    forward. Keeps the sampler agnostic to per-paper conditioning.

## Things I would peek at GitHub for (if allowed)

The blind-impl constraint flagged these as the highest-value diffs to
learn from in Phase 2:

- **Exact PINN PDE form** — diffusion coefficient ν, whether the residual
  is a true transient PDE (∂I/∂t = ν ∇²I) integrated along the diffusion
  timesteps or a steady-state form (decisions 12 / 13 / 14 / 15).
- **Stroke-order encoder** — vector pooling (ours) vs cross-attention
  Transformer (decision 8 / 10).
- **Multi-attribute fusion** — sum + MLP (ours) vs concatenation,
  AdaLN-Zero, hypernetwork (decision 7).
- **Content cache shape** — paper does not document its inputs; we assume
  bitmap (Stage A) and bitmap+skeleton (Stage B/C) (decision 27).
- **Exact U-Net widths / depths / attn placement** (decisions 3, 4, 5).
- **PINN weights and stage schedule** (decisions 13, 14).
- **Whether the model uses EMA on diffusion weights** (we do not).
- **The β schedule and T** (decisions 17, 18, 19).
- **The stroke vocabulary and source DB** (decision 9).

## Open questions / known limitations

1. The stroke-order, ink-intensity, and font-size fields are *placeholders*
   synthesised from a hash of char + writer. Real Stage B/C runs need to
   swap them with a stroke-order DB lookup and the actual Ernantang
   per-row metadata before being meaningful.
2. The PINN PDE is a *surrogate* of the paper's intent — three
   differentiable physical priors that *plausibly* match the paper's
   description. They are not derived from a published equation.
3. EMA on diffusion weights is not implemented. Add 0.9999 decay before
   the Stage B handover.
4. Cross-language eval (CN -> KR) — out of Phase 1 scope.
5. Real Stage B / C training has not been launched; the configs point at
   smoke manifests so dry-run does not require the full snapshot.

## Reproducibility checklist (per rubric)

- [x] `torch.manual_seed`, `numpy.random.seed`, `random.seed` set in
      `train.main`.
- [x] No hardcoded SSH credentials or absolute internal paths in `.py`.
- [x] `--device` is honored, defaults to `cuda:0`, not hardcoded.
- [x] No silent `except: pass` in the package.
- [x] All paths resolve through `paper_reimpl_shared.data.manifest`.
- [x] `tests/test_smoke.py` exercises forward + backward + 1 optimizer
      step + PINN gradient + 1 sampling pass.

## Verification commands

From `papers/08_dp_font/`:

```
uv venv --python 3.11
uv sync                                         # installs deps + builds editable shared
uv run ruff check src tests
uv run pytest tests/test_smoke.py -x -v

uv run python -m paper_reimpl_shared.runner.entrypoint \
    --paper dp_font --dry-run --synthetic --device cpu \
    --train src/dp_font/configs/train_stage_a_ttf.yaml \
    --model src/dp_font/configs/model.yaml \
    --data src/dp_font/configs/data_stage_a.yaml \
    --data-backend mac_symlink
```

Expected: smoke tests pass; dry-run prints 1 finite `loss_simple` and
returns code 0.
