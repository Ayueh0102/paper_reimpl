# 07 — Moyun blind impl decision log

Blind worker for Liu et al. 2024 *Moyun*, ACM McGE 2025. No GitHub URL is
known for this paper (`facts_code_url: null`). All decisions below come from
the paper notes (`/Users/Ayueh/Documents/Obsidian Vault/research/papers/Zhang_2024_Moyun_DiffusionMambaCalligraphyGeneration.md`)
plus first-principles reasoning. Decisions are tagged as either:

- `[paper-cited <section>]` — directly stated in the paper / note
- `[guessed-because-paper-vague]` — paper does not specify; I picked a value

## Architectural decisions

### A1. Backbone arch [paper-cited §3.3]
Vision Mamba (Mamba2) replaces U-Net. Paper §3.3 pins: VAE latent 32×32×4,
patchify with `patch_size=8`, `hidden_dim=512`, `N=4` Mamba blocks. These are
all carried into `MoyunConfig` defaults verbatim.

### A2. Mamba kernel backend [guessed-because-paper-vague — infra fallback]
The paper says "Mamba2" but does not specify the implementation. The
reference CUDA kernel (`mamba-ssm`) needs `nvcc` + Triton and:
- Does not build on macOS (no CUDA).
- Wheels are sparse on Windows; source build needs MSVC + matching CUDA.

Since this blind reimpl must `uv sync` cleanly on the Mac dev box AND run on
both Mac (CPU smoke) and Windows PC (3090 training), I ship a **pure-PyTorch
S6 (Mamba1) sequential scan** in `src/moyun/mamba_block.py`. This is
algorithmically equivalent to Mamba1, slightly slower than the fused kernel
at large `L`, but correct by construction. The interface is a thin wrapper so
swapping to `mamba_ssm.selective_scan_fn` later is one `if` branch.

**Risk**: at our paper-scale `L = (32/8)^2 = 16`, the sequential scan is
fast even on CPU. At larger L (e.g. unpatchified pixel-space Mamba) the
slowdown grows linearly. We do not pay that cost here.

### A3. Mamba2 vs Mamba1 [guessed-because-paper-vague]
The paper specifically calls out "Mamba2" but does not give the SSD-specific
restrictions (scalar-per-head `A`, structured matmul). I implement the
algorithmically equivalent **per-channel diagonal `A`** form (Mamba1 / S6).
This gives the same recurrence; only the GPU kernel structure differs. For
the blind reimpl this is the same math.

### A4. SSM hyperparams `d_state`, `d_conv`, `d_dt_rank` [guessed-because-paper-vague — Mamba1 defaults]
Paper does not specify any of these. Defaults from the original Mamba paper
(Gu & Dao 2024, §3.4): `d_state=16`, `d_conv=3`, `d_dt_rank=ceil(d_model/16)`.
Community-validated as safe; exposed as `MoyunConfig` knobs so we can
ablate later.

### A5. Bidirectional scan [guessed-because-paper-vague — Vision Mamba 2024 default]
Paper does not say whether the SSM is run unidirectionally or both
forward+reversed. I default to bidirectional (Zhu et al. 2024). A
unidirectional 2-D patch sequence has well-known asymmetry artifacts; the
extra cost of one more scan per block is acceptable at our scale.

### A6. Positional embedding [guessed-because-paper-vague]
Paper does not specify. I use a **learnable 1-D positional embedding** of
length `(image_size / patch_size)^2`. DiT (Peebles & Xie 2023, §3.1) uses
2-D sinusoidal; that would be a small ablation but unlikely to change
results at our scale.

### A7. TripleLabel embedding layout [paper-cited §3.4]
Three INDEPENDENT `nn.Embedding` tables (writer / script / char), summed
into `e_total`, then fed to a `Linear -> SiLU -> Linear` MLP that produces
per-block `(scale_ssm, shift_ssm, scale_ffn, shift_ffn)`. The DL review
rubric explicitly calls out "the three embeddings must each be independent"
— this is enforced by code (distinct `nn.Embedding` modules) and by
`tests/test_smoke.py::test_triple_label_embeddings_are_distinct_modules`.

### A8. CFG `[NULL]` token encoding [guessed-because-paper-vague]
Paper says CFG is used but does not define the "uncond" branch encoding. I
reserve **index 0** of each embedding table as a `[NULL]` row, zero-initialized.
`Moyun.forward` shifts user-supplied ids by +1 internally so the real id
range starts at 1. `cfg_drop_prob` randomly maps a sample's labels to -1
(which becomes 0 after the shift). This is the simplest scheme that keeps
`writer_id=None` (sampler-time uncond) and `writer_id=<id>` (training-time)
on the same code path.

### A9. Label dropout mode [guessed-because-paper-vague]
Paper §3.5 mentions "classifier-free guidance" but does not say whether the
three labels are dropped jointly (one Bernoulli per sample) or independently
(three Bernoullis per sample). I default to **joint** because that is the
standard CFG recipe (Ho & Salimans 2022) and it gives a cleaner conditional
vs unconditional split. `independent` is exposed via `label_dropout_mode`
config knob for ablation — independent dropout would let the model also
generate "writer=A, script=?, char=B" partials.

### A10. adaLN-Zero init [guessed-from-DiT §3.2]
The conditioning MLP's final linear is **zero-initialized** so each block
starts as an identity time-mixer (DiT §3.2). Paper §3.4 says "DiT-style
scale-shift modulation" which implies the DiT convention; I follow it. Note:
this means the TripleLabel embedding gradients are zero on the very first
iteration — the smoke test handles this by running TWO backward passes
(first one trains the modulation MLP off zero; second one shows real
embedding gradients).

### A11. Time embedding dim [guessed-because-paper-vague]
Paper does not specify the timestep embedding width. I use
`time_embed_dim = hidden_dim = 512` and a 2-layer SiLU MLP. This matches
DiT and is overkill for `T = 1000` but cheap.

### A12. VAE [paper-cited §3.3 / not implemented]
The paper uses a pretrained VAE to compress 256×256 images to 32×32×4
latents. Which VAE is not specified. **We do not ship a VAE** — the model
accepts `in_channels=4` (latent mode) or `in_channels=1` (direct pixel
mode, used by the smoke test). Production training will need either:
1. Fine-tune SD-1.5 VAE on Ernantang renders (recommended — its latent
   priors are wrong for binary glyphs out of the box).
2. Train a small bespoke VAE on TTF renders as Stage-0.
Either way, this is upstream infra that the paper assumes exists.

### A13. Loss formulation [paper-cited §3.5]
Standard DDPM denoising MSE (`F.mse_loss(model_pred, ε_true, reduction='mean')`).
Paper does not mention any auxiliary loss term. CFG is achieved via input
dropout only, not via a separate loss head.

### A14. Diffusion target [guessed-because-paper-vague]
Paper just says "DDPM". I default to `prediction_target='epsilon'` because
that is the DDPM paper (Ho et al. 2020) default. `x0` and `v` prediction
are both possible — the shared `GaussianDiffusion` supports the toggle, and
this is an easy ablation.

### A15. β schedule [guessed-because-paper-vague]
Paper does not state. I use **linear** β from 1e-4 to 2e-2 over 1000 steps
(DDPM defaults). Cosine schedule (Nichol & Dhariwal 2021) is a known small
improvement; held as a future ablation.

### A16. Optimizer + schedule [paper-cited §3.5 / guessed]
- `AdamW`, `lr = 1e-4`: `[paper §3.5]`.
- `(β1, β2) = (0.9, 0.999)`, `weight_decay = 0.0`, `grad_clip = 1.0`:
  `[guessed]` standard DDPM/DiT recipe.
- LR schedule: no decay in Stage A; cosine warmup left for production tuning.

### A17. Batch / steps [paper §3.5 partial]
Paper: global batch 768 across 3× A100, 288 000 iterations. We scale down
to `batch=16` and `max_steps=5000` for Stage A on single 24 GB GPU. This is
a `[guessed]` scale-down — linear-rule lr would say to lower the lr to
`1e-4 * 16/768 ≈ 2e-6` but DDPM is known to be relatively robust to batch
scaling so we keep `lr=1e-4` and let the trainer accumulate signal over
more steps. Production tuning may need a sweep here.

### A18. Manifest backend [policy — AGENTS.md §0.4]
All data paths go through `paper_reimpl_shared.data.manifest.BackendPaths`;
no path is hardcoded in any `.py` file. Smoke / dry-run uses the synthetic
dataset; Stage A/B/C consume manifest JSONLs resolved by `--data-backend`.

### A19. Sampler [paper-cited §3.5 / guessed schedule]
Paper §3.5 says "DDPM" sampler for inference. We expose both DDPM and DDIM
via the shared sampler; `MoyunSampleConfig.sampler` defaults to DDPM.
Number of inference steps is not specified by the paper — `[guessed]` 1000
full DDPM or 50 DDIM are both fine.

### A20. Sample-time CFG mechanics [paper-cited / shared-sampler integration]
The shared sampler's `cfg_uncond_drops_content=False` flag is passed
through `moyun.sample.sample` — Moyun has no content path, so the uncond
branch is constructed by passing `writer_id=script_id=char_id=None` (which
the model routes to the `[NULL]` embedding rows). This avoids the
FontDiffuser-style "zero out content" hack.

## Mamba install fallback — detailed plan

Since the paper note explicitly flags this concern, here's the contingency:

1. **Mac dev**: pure-PyTorch scan only. `mamba-ssm` is not installable. CPU
   smoke test runs in ~1 second.
2. **Linux GPU box (e.g. 3090 / RTX 6000 Ada)**: try `uv add mamba-ssm` first.
   If that fails (it often does on first attempt because of nvcc version
   mismatch), fall back to pure-PyTorch scan. The model trains identically;
   only throughput differs. Expected gap: ~2-4× slower per Mamba block at
   `L = 16`, ~10× slower at `L = 256`. At `L = 16` (paper config) the gap
   barely matters.
3. **Future fast-path swap**: in `mamba_block.py`, replace
   `SelectiveScanSSM.forward`'s sequential loop with a call to
   `mamba_ssm.selective_scan_fn(...)`. The pre/post projections stay the
   same. Single-file change.

The pure-PyTorch scan has been validated by `tests/test_smoke.py` —
gradient flows through `A_log`, `D`, `in_proj`, `conv1d`, `x_to_BCdt`,
`dt_proj`, `out_proj`, the bidirectional `ssm_b`, the per-block FFN, the
TripleLabel embeddings, and the cond_mlp.

## Open questions for Phase 2 github-diff

1. Does the official repo use SD-1.5 VAE, a custom VAE, or no VAE at all?
2. Mamba1 vs Mamba2: the paper says Mamba2 but does it actually use
   `mamba_ssm.modules.Mamba2`? If the official uses Mamba1 our blind impl
   is closer to ground truth than the paper text suggests.
3. What is the actual `d_state`?
4. Is the SSM bidirectional or unidirectional?
5. Is `cfg_drop_prob` joint or independent?
6. Is the prediction target ε or x0 or v?
7. Is the β schedule linear, cosine, or scaled-linear?
8. Is the positional embedding learnable, 1-D sinusoidal, or 2-D sinusoidal?
