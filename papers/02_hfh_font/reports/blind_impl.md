# 02_hfh_font — blind-implementation decision log

Phase-1 reimpl-worker decisions, tagged by source. ``[paper-cited <ref>]`` means
the project paper note (``021_HFH-Font...md``) or the paper PDF mentioned it
explicitly. ``[guessed-<reason>]`` flags every interpolation the worker had
to make because the source was silent.

## Architecture

- **[paper-cited §訓練配置] Latent diffusion @ 64×64 with VAE 8× downsample.**
  Implemented as ``TinyVAE(down_factor=8)`` with grayscale in/out. The exact
  SD VAE variant is *not* named in the paper note.
- **[paper-cited §核心方法論] Cross-attention conditioning over component-level
  reference features.** Implemented as ``ComponentEncoder`` emitting
  ``N_refs * K_comp`` tokens → ``_CrossAttention`` mid-block in
  ``LatentUNet``. ``K_comp = 4`` (a 2×2 adaptive pool) is a [guessed] default
  because the paper does not state the per-reference token count.
- **[guessed-because-paper-vague] U-Net depth = 3 levels (base × {1,2,4}),
  2 residual blocks per level, base_channels=64.** The note does not give
  any depth/width numbers. This is a memory-conservative shape relative to
  SD-1.5 (base=320) for our single-RTX-6000-Ada setup.
- **[guessed-because-paper-vague] AdaLN-Zero (DiT-style) modulation for
  time + char + writer + script.** The note never says the modulation
  scheme; AdaLN-Zero is chosen because it (a) is well-understood, (b)
  zero-init at module entry guarantees the network starts from identity,
  helping smoke-test stability.
- **[guessed] Content image is concatenated to ``z_t`` after spatial
  downsample.** The paper note does not say how the content path is fed.
  Concat-along-channels is the simplest defensible plumbing; alternatives
  (cross-attention from content, AdaLN-from-content) deferred to Phase-2.
- **[guessed] ``components_per_ref = 4`` (2×2 token grid per ref).** The paper
  claims "component-level" tokens; without IDS-derived component count per
  char we use a fixed grid pool. Increasing to 8/16 is a Phase-2 ablation.
- **[guessed] Cross-attention applied only at the U-Net mid block.**
  ``attention_resolutions`` config is honored but only mid-attn is wired in
  the current scaffold. Extending to every down/up level is a Phase-2 hook.

## Loss

- **[paper-cited Plan-A convention] ``prediction_target = x0``.** The paper
  note does not state ε vs. x0; we align with the project-wide Plan-A
  convention (``shared.diffusion.GaussianDiffusion``). Epsilon prediction
  remains a config switch.
- **[guessed-because-paper-vague] SDS loss = MSE between student 1-step x0
  prediction and teacher x0 prediction at random ``t``.** This is a
  *placeholder* for the formal SDS gradient:
  ``∇_θ_S L = E_t[ w(t)(θ_T(z_t,t) − ε) · ∂z_t/∂θ_S ]``. Phase-2 will
  swap in the full form if needed.
- **[paper-cited §訓練配置] CFG dropout p̂ = 0.1.** Applied uniformly to
  char / writer / script / ref channels. The note actually applies p̂ only
  to "same-char ref replacement in the last 10% of iters" — our broader
  application is a [guessed-conservative] extension to enable inference-time
  CFG on every conditioning channel.
- **[guessed] L_SR = L1 only.** No LPIPS / perceptual term. The paper does
  not detail the SR loss formulation; this is the cheapest defensible
  baseline.
- **[paper-cited §訓練配置] CFG scale sc = ss = 2.0** baked into the
  Stage-C train YAML (``cfg_scale: 2.0``).

## Diffusion schedule

- **[guessed-because-paper-vague] Training horizon T = 1000.** The paper
  only names the *inference* step count (10) with trailing timestep
  selection. T=1000 is the DDPM default.
- **[guessed] β schedule = linear (β1=1e-4, βT=2e-2).** Cosine remains a
  config switch.
- **[guessed] "Trailing" 10-step inference approximated by 10-step DDIM.**
  We do not implement the exact trailing timestep formula because the note
  does not provide it. ``GaussianDiffusion.sample(sampler="ddim")`` runs the
  same number of inference steps; the timestep spacing differs but the
  smoke-test only validates shape + finiteness.

## Training schedule

- **[guessed-from-table] Stage A batch_size = 16, lr = 1e-4.** The paper
  used batch 64 (small) / 128 (large) on a single A100. Linear-scaled to
  one RTX-6000-Ada we'd run batch ≈ 32; we go further down to 16 for the
  Phase-1 scaffold so that 24+ GB VRAM is reserved for the U-Net depth and
  cross-attention.
- **[guessed] Stage B lr = 5e-5, Stage C lr = 2e-5.** Each subsequent stage
  drops lr by ~2-5×. The note does not specify per-stage lr values.
- **[guessed] AdamW weight_decay = 0.01, β = (0.9, 0.999).** Standard
  diffusion defaults.
- **[guessed] grad_clip = 1.0.** Standard.

## Data

- **[paper-cited §訓練配置] image_size = (LR 512 → HR 1024)**. The scaffold
  uses 128 for smoke tests; data configs default to 128 because the
  ``content_fields_cache`` in the mother repo is 128. Stage-B/C data
  configs can override.
- **[guessed] content channels = [bitmap, sdf, skeleton].** Matches the
  project's Plan-A default. The paper does not say what its content
  representation is.
- **[guessed] n_refs = 4.** The paper supports variable counts (few-shot
  to mid-shot). Four matches the project-wide default.
- **[guessed] Reference glyph dropout: zero out the entire ref stack when
  CFG drops apply.** Alternative (per-token mask) deferred.

## SR module

- **[guessed-entire-module] Style-guided SR = a tiny U-Net stem + cross-attention
  + ``PixelShuffle`` upsample by 2×.** The paper says only "style-guided
  super-resolution module" — zero arch detail. Our impl is a placeholder;
  Phase-2 will diff against any official source.
- **[paper-cited] SR is the final stage, initialised from low-res weights.**
  We expose this as ``sr_enabled: false`` by default; the SR module is
  *built but unused* until Stage-D (not yet ship-able).

## Things explicitly skipped in Phase-1

- Pretrained VAE checkpoint. The ``TinyVAE`` ships as an *unfrozen* random
  init for smoke testing. A pretrained VAE swap-in must come before any
  real training.
- IDS-driven component decomposition. The hook (per-ref token count) is
  there; the lookup integration to ``~/Char/datasets/ids/`` is not. Adding
  it is the natural Phase-2 follow-up if reviewers say the abstract
  "component" pool is insufficient.
- LPIPS / FID / Acc(C) / Acc(S) eval metrics. Out of Phase-1 scope.
- 1024×1024 generation end-to-end. SR is built-but-disabled.

## Suspected weaknesses

1. **VAE is not pretrained**: training without a pretrained VAE will likely
   not converge to glyph-quality reconstruction. Stage-A must include a
   VAE warmup phase or load a pretrained checkpoint before the latent
   diffusion loss is meaningful.
2. **Component encoder is a vanilla CNN**: the paper claims component-level
   reuse, but our encoder has no inductive bias for components. If the
   official architecture exposes a parsing-aware front-end, our diffs in
   Phase-2 will be large.
3. **SDS loss is a placeholder MSE**: the real SDS gradient is *not* the
   same thing as MSE between student and teacher x0; reviewers will
   correctly flag this.
4. **Trailing timestep sampler is approximated**: 10-step DDIM is not the
   same as trailing-10 DDPM. Phase-2 must fix this if FID matters.
5. **CFG dropout applied uniformly to all cond channels**: paper only
   specifies p̂=0.1 on style refs in the last 10% of iters. Broader
   application may degrade learning of the strong conditioning signals
   (char_id especially).
