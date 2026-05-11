# QT-Font — Phase 2 GitHub Diff

`STATUS: official_available`

- **Official repo**: https://github.com/lsflyt-pku/QT-Font (cloned via `git clone --depth=1` to `third_party/05_qt_font/`)
- **Phase 0 row**: `reports/phase0_spec_table.md:112` — `github_url: null` (we recovered it in Phase 2 via Google + PapersWithCode-style search).
- **Phase 1 blind notes**: `papers/05_qt_font/paper_notes/05.md`, decision log `papers/05_qt_font/reports/blind_impl.md`.
- **Reviewer scope**: this report classifies every non-trivial divergence into the seven AGENTS-defined buckets. No code is modified — fixes are deferred to the reimpl-worker.

A one-line headline before the detailed diff: **our blind impl and the official repo share almost no implementation surface**. The paper's "quadtree representation" is implemented in the official code as a **2-D port of the OCNN/DualOctreeGNN library on a point cloud of stroke+skeleton points** (`third_party/05_qt_font/ocnn_2d/octree/octree.py:161`), with discrete diffusion over **K=3** node states (`empty / stroke-edge / skeleton`) and a **graph U-Net with depth-typed message passing on a dual graph of leaf-octree + internal-octree nodes** (`third_party/05_qt_font/models/graph_diffusion.py:47`). The blind impl is a **fixed-depth saturated raster quadtree with K=8 generic intensity bins and a plain 4-connectivity GraphConv** (`papers/05_qt_font/src/qt_font/model.py:43-176`). Most of the deltas below are direct consequences of that single representational chasm.

---

## 1. arch_deltas

| # | Topic | Blind impl (`papers/05_qt_font/src/qt_font/...`) | Official (`third_party/05_qt_font/...`) | Severity |
|---|---|---|---|---|
| A1 | **Tree topology** | **Full saturated** quadtree at fixed `depth=4` → every sample has `4^4 = 256` leaves (`src/qt_font/model.py:43`, `model.py:109-133`). | **Adaptive sparse** octree-style tree built from a stroke point cloud via `ocnn_2d` (`ocnn_2d/octree/octree.py:161-240`). Layers `[0, full_depth]` are dense (default `full_depth=4`); layers `(full_depth, depth]` are sparse and reuse OCNN's `xyz2key` / shuffled-key Morton scheme (`octree/octree.py:185-225`). | **CRITICAL** — already flagged `[guessed-major]` in `reports/blind_impl.md:55-65`. This is the largest single deviation; the paper's whole "O(N) instead of O(L²)" claim rests on this. |
| A2 | **Tree depth** | `depth=4` (256 leaves on 16×16 grid) (`configs/model.yaml:13`). | `depth=7` (128×128 leaf grid, sparse), with `full_depth=4` (`configs/chinesefont_train.yaml:44-46`). 256/512 px use `depth=8/9`. | **HIGH** — blind impl is **3 levels shallower** at the official 128 px target, so leaf count differs by 4³ = 64×. |
| A3 | **Leaf state cardinality `K`** | `n_states=8` categorical bins ([3-bit intensity], `configs/model.yaml:14`, `blind_impl.md:67-71`). | **`num_classes=3`** — three classes representing `{background, stroke-contour, skeleton}` (`main.py:156`, `datasets/chinesefont_asymmetric.py:35`). The third class is harvested from `morphology.skeletonize()` (`datasets/chinesefont_asymmetric.py:305`). | **CRITICAL** — paper does not use generic intensity bins; the K=3 labels are derived from `cv2.findContours()` + `skimage.morphology.skeletonize` and constitute the "axis" supervision (cf. `losses/loss.py:25` `label_gt = F[:, 2] * 1 + F[:, 3] * 2`). |
| A4 | **Graph layer family** | Custom additive GraphConv on a 4-connectivity sibling grid (cf. `model.py` `_build_leaf_sibling_index`). | OCNN's **`GraphConv` with depth-typed weight matrix of shape `(n_edge_type * (in+node_ch), out)` and 5 directional edge types per node, plus `n_node_type` per-depth weight slice** (`models/modules_bn.py:66-110`). Residual block is `GraphResBlocks` with per-block timestep + cond conditioning (`graph_diffusion.py:85`). | **HIGH** — the official message passing is **edge-direction-aware (N/E/S/W + self = 5)** and **node-type-aware** (one weight per octree depth); the blind impl has neither. |
| A5 | **Dual graph mechanism** | Sequential `fine → ContentAwarePool → coarse → broadcast` (`paper_notes/05.md:111-119`). | **U-Net over leaf-octree (encoder) + create-on-the-fly internal-octree (`create_full_octree`) (decoder)**; skip connections cross between encoder and decoder graphs through `doctree_align` (`graph_diffusion.py:134, 199-201`). "Dual" here = **two octrees** (input/output), not two adjacencies. | **CRITICAL** — interpretation gap. Blind impl's "dual" is a 2-stage pyramid; official's "dual" is an asymmetric U-Net between an in-octree and an out-octree that **grows** during decoding (`graph_diffusion.py:222-232`). |
| A6 | **Content-aware pooling** | Softmax attention over each parent's 4 children via small MLP (`paper_notes/05.md:114-117`). | No standalone "content-aware pool" module by name. The functional equivalent is OCNN's `GraphDownsample` (channels-aware, depth-typed) (`graph_diffusion.py:74, 80, 87`), plus an `'align' in cond` decoder skip that re-aligns leaf keys with `doctree_align`. The "content awareness" is realised at the **decoder-skip-from-content-octree** path (`graph_diffusion.py:198-201`). | **HIGH** — blind impl invented an attention pool that the paper does not have; the real "content-aware" trick is decoder-side feature reuse from a separately encoded content octree. |
| A7 | **Predict-and-grow head** | Single per-leaf MLP `→ (B, L, K)` at the deepest level only (`paper_notes/05.md:120`). | **Per-depth `predict` heads** (`graph_diffusion.py:112-113`); at every decoder depth `d ∈ [depth_stop, depth_out]` the model emits a logit and **splits the octree on-the-fly** via `octree_split` / `octree_grow` (`graph_diffusion.py:222-232`). Predict head emits **2 channels at inner levels, 3 channels at `depth_out`** (the leaf level). | **CRITICAL** — blind impl only predicts at the leaf level; the official model predicts splitting probabilities at every level and grows the output octree progressively. The loss is summed over levels (`losses/loss.py:43-47`). |
| A8 | **Style/content encoder** | 3-conv CNN over the reference / content **image**, mean-pool over `R` refs, additively summed into the conditioning vector (`paper_notes/05.md:106-110`). | Both content **and** style encoders are themselves **graph U-Net encoders over per-reference octrees** built from points-on-contour: `style_conv + style_encoder + style_downsample` (`graph_diffusion.py:69-80`). Per-ref features are projected to a global vector by `(maxpool ⊕ avgpool) → Linear`, then averaged across refs (`graph_diffusion.py:316-322`). | **CRITICAL** — style is encoded **in the same quadtree space**, not as a CNN feature. This is what makes "few-shot via geometric similarity" work in the paper. |
| A9 | **Conditioning injection** | Additive into a single `cond` vector that is broadcast over every leaf (`blind_impl.md:97-103`). | **Per-depth `temb = timesteps + cond` injected into every `GraphResBlock`** at every depth, with `batch_id` indexing recovered from Morton-key prefix (`graph_diffusion.py:148-149, 162-163, 186-187, 208-209`). Style + content cat to width `2 * channels[depth_stop]`, then split via `style_fc` / `content_fc` if both 'char' and 'font' are in `cond` (`graph_diffusion.py:116-119, 354-380`). | **HIGH** — additive-broadcast is correct in spirit but **at the wrong granularity**: the official injection is per-resblock per-depth, not once at input. |
| A10 | **Time embedding** | Sinusoidal (base 10 000) + 2-layer MLP (`paper_notes/05.md:128`). | Sinusoidal (base 10 000) + 2-layer MLP via `TimeEmbedding(n_channels)` with `Swish` activation (`graph_diffusion.py:26-44`). | **LOW** — broadly equivalent; activation differs (Swish vs GELU). |

---

## 2. loss_deltas

| # | Topic | Blind impl | Official | Severity |
|---|---|---|---|---|
| L1 | **Loss form** | D3PM-style CE on per-leaf K-way logits vs. integer `x₀`, evaluated **only at the leaf level** (`train.compute_loss`, `paper_notes/05.md:82`). | **Multi-depth cross-entropy**: `axis_loss` emits one `F.cross_entropy(logits[d], label_gt)` per decoder depth `d` and the trainer sums them (`losses/loss.py:14-31`, `main.py:60-62`). At the leaf level `d == max_depth` the label is the **3-class axis label** `F[:, 2]*1 + F[:, 3]*2`; at inner levels it is the **`nempty_mask`** (split/no-split). | **CRITICAL** — paper trains a **structural-split classifier at every level**, plus a 3-class axis classifier at the leaf. Blind impl has neither — it only has one CE at the leaf level over `K=8` intensity bins. |
| L2 | **Loss reduction / weighting** | Single CE term, default mean reduction (`F.cross_entropy`). | `compute_octree_loss` includes a per-depth weights vector `weights = [1.0]*16`, with a commented-out down-weighting schedule `[1.0]*4 + [0.8, 0.6, 0.4] + [0.2]*16` (`losses/loss.py:39-40`). Currently every depth is weight 1.0. | **MEDIUM** — small, but the existence of the weighting vector flags a known training-stability tuning knob. |
| L3 | **Auxiliary regularisers** | None beyond the single CE (blind_impl §25). | The active loss is `axis_loss` only (no SDF, no occupancy term). `compute_loss` sums every entry whose key contains `'loss'` (`main.py:60-62`), and the model out also exposes `accu_d` accuracy diagnostics. | **LOW** — concept gap; blind impl skipped a multi-depth loss the paper explicitly uses, not an extra regulariser. |
| L4 | **`pos` / `grad` plumbing** | Not present. | `batch_to_cuda` calls `batch['pos'].requires_grad_()` (`main.py:47`), and dataset emits `pos` (`datasets/chinesefont_asymmetric.py:218`). No gradient-based loss is currently active against `pos`, but the hook is wired in case a gradient-norm regulariser is enabled. | **LOW** — dead code in current config, but worth noting in case Stage B/C reactivates it. |

---

## 3. schedule_deltas

| # | Topic | Blind impl | Official | Severity |
|---|---|---|---|---|
| S1 | **`T` (num timesteps)** | `timesteps: 100` (`configs/model.yaml:25`, `configs/train_stage_a_ttf.yaml:23`). | **`num_train_timesteps: 1000`** (`configs/chinesefont_train.yaml:65, 93`; hard-coded again at `main.py:155`). | **HIGH** — `T` is **10× larger** than blind default. Direct consequence of the `[guessed]` in `blind_impl.md:81-83`. |
| S2 | **β schedule** | **Linear** `[1e-4, 0.02]` (`configs/train_stage_a_ttf.yaml:24-25`). | **Cosine (Glide)**: `beta_schedule: 'cos'`, `α̅_t = cos²((t/T + 0.008) / 1.008 · π/2)`, then `β_t = min(1 - α̅_t/α̅_{t-1}, 0.999)` (`configs/chinesefont_train.yaml:66`, `datasets/chinesefont_asymmetric.py:55-60`). `beta_start: 0.02`, `beta_end: 1.0` exist but are **only used by `linear`/`scaled_linear`/`sigmoid` branches**, not by `'cos'`. | **HIGH** — schedule shape difference + cosine clipping. With T=1000 + Glide-cosine the early-noise schedule is much gentler than linear-100. |
| S3 | **Q matrix form** | Uniform `Q_t = (1-β_t)·I + (β_t/K)·11ᵀ` (`blind_impl.md:30-32`). | **Same family, but with K=3 and the diagonal correction `1 - β_t·(K-1)/K`** instead of `1 - β_t`: `Q[t, i, i] = 1 - β·(K-1)/K`, `Q[t, i, j≠i] = β/K` (`main.py:171-177`, `datasets/chinesefont_asymmetric.py:68-74`). Both are valid uniform-D3PM parameterisations; the official form keeps the row sum exactly 1 for **any** K. | **MEDIUM** — semantically equivalent up to a `(K-1)/K` factor in `β`; numerically different. Blind form has row-sum drift unless K is treated specially. |
| S4 | **Sampler — reverse process** | "x₀-conditional, single multinomial, argmax on `t=0`" (`paper_notes/05.md:188-200`). | **Full D3PM posterior** with Gumbel-max sampling: `p(x_{t-1} | x_t, x₀)` is computed via `q_posterior_logits` (`main.py:215-244`) and `p_sample` (`main.py:258-273`); `gap=50` is the **stride** (T=1000 / 50 = 20 effective denoising steps), and on-the-fly per-step Q is recomputed when `gap != 1` (`main.py:225-229`). | **HIGH** — official uses **fewer effective steps (20)** via stride sampling, not 100 dense steps. The reverse model also iteratively **rebuilds the input octree** every step from the sampled label image (`main.py:298-...`). |
| S5 | **`lr`, optimizer** | AdamW, `lr=1e-4`, wd `1e-2`, `cosine` (`configs/train_stage_a_ttf.yaml:13-15`). | AdamW, `lr=0.0001`, **`weight_decay=0.0`** (`configs/chinesefont_train.yaml:25-27`), `lr_type: cos`, `step_size: (160, 240)`. | **MEDIUM** — blind impl uses **`wd=0.01`** but paper config uses **`wd=0.0`**. Hyperparam, not critical, but it's an explicit deviation against `[paper-cited 訓練配置]` in `blind_impl.md:88-90` (the note's claim of wd=1e-2 is at minimum inconsistent with the official YAML). |
| S6 | **Effective batch / accum** | `batch_size=8`, no accumulation (`configs/train_stage_a_ttf.yaml:11`, `blind_impl.md:147-149`). | `batch_size=4` per GPU, **`accum=32`**, **8 GPUs** → effective 1024 (`configs/chinesefont_train.yaml:28, 59`). | **HIGH** — official trains with **128× the effective batch** of blind impl. |
| S7 | **Epochs / steps** | `max_steps=16000`, `max_epochs=1` (`configs/train_stage_a_ttf.yaml:16-17`). | `max_epoch=20`, `test_every_epoch=1`, `log_per_iter=50` (`configs/chinesefont_train.yaml:16-18`). The "≈ 16k steps" estimate in `paper_notes/05.md:144-145` was a back-of-the-envelope. | **MEDIUM** — the blind step count is roughly compatible *if* and only if the per-epoch iteration count of the official dataset is ≈ 800. With unknown dataset cardinality it could be off by 5× either way. |

---

## 4. conditioning_deltas

| # | Topic | Blind impl | Official | Severity |
|---|---|---|---|---|
| C1 | **Conditioning modes** | `char_id`, `writer_id`, `script_id` embeddings + 1 content image + R reference images (`paper_notes/05.md:128-140`). | A single string `cond` flag: `{'char', 'font', 'char_font', 'char_font_align'}` (`graph_diffusion.py:64, 354-380`). The training config uses `cond: char_font_align` (`configs/chinesefont_train.yaml:114`). **No `char_id` / `writer_id` / `script_id` embeddings** — style is image (octree) only, content is image (octree) only. | **CRITICAL** — blind impl invented categorical id embeddings the paper does not use. CFG dropout on those embeddings is therefore not equivalent to the paper either. |
| C2 | **Style path** | 3-conv CNN + mean-pool over R refs, `R=1..4` (`paper_notes/05.md:106-110`). | **Per-ref graph encoder over a points-on-contour octree** (`graph_diffusion.py:69-74, 239-272`); each ref independently encoded then averaged across refs in feature space (`graph_diffusion.py:316-322`). `ref_num=3` per `configs/chinesefont_train.yaml:50`. | **HIGH** — encoder family mismatch (CNN vs octree graph U-Net). |
| C3 | **Content path** | 3-conv CNN over a content bitmap (Stage A: 1 ch; Stage B/C: 2/3 ch with bitmap+sdf+skeleton) (`paper_notes/05.md:106-110`). | Single graph encoder over a points-on-contour octree for the source-font character; uses **stroke contours only** (no SDF, no skeleton, no IDS) (`datasets/chinesefont_asymmetric.py:303-309`). `c_ref_num=1` (`configs/chinesefont_train.yaml:51`). | **HIGH** — blind impl invented multi-channel content the paper does not use. |
| C4 | **Conditioning fusion** | All paths → sum into a single cond vector → broadcast over every leaf (`blind_impl.md:97-103`). | `cond = concat(content_feature, style_feature)` (`graph_diffusion.py:369`); `cond_for_resblock = timesteps + cond[batch_id]` injected at **every resblock at every depth** (`graph_diffusion.py:148-149`). | **HIGH** — different fusion + different injection granularity. |
| C5 | **`'align' in cond`** | No analogue. | Decoder skip from content-octree's encoder features keyed by `doctree_align` (`graph_diffusion.py:198-201`). This is the closest thing to a "content-aware" mechanism in the paper. | **HIGH** — major missing feature in blind impl. |
| C6 | **CFG dropout** | `cfg_drop_prob=0.1`, drops categorical ids only (`configs/train_stage_a_ttf.yaml:19`). | **No CFG / classifier-free guidance found** in either training (`main.py`) or sampling (`main.py:284-...`). The `'cond'` string toggles whole pathways at model-construction time, but there is no per-batch dropout. | **HIGH** — blind impl added CFG; paper appears not to use it. Removing it for parity is fine but may hurt sample diversity. |

---

## 5. hparam_deltas

| # | Topic | Blind impl | Official | Severity |
|---|---|---|---|---|
| H1 | **Channels** | `hidden_dim=128`, single width (`configs/model.yaml:17`). | **Per-depth widths**: `self.channels = [3, 512, 512, 256, 512, 256, 128, 64, 64, 64]` (128/256 px) (`graph_diffusion.py:421`). The 512-wide block at `depth_stop=4` is the bottleneck (`configs/chinesefont_train.yaml:104-105`). 512 px variant uses `[3, 512, 512, 256, 512, 256, 256, 128, 128, 64]` (`graph_diffusion.py:422`). | **HIGH** — channel pyramid is materially wider than blind impl. |
| H2 | **Resblock layout** | `n_layers: 3` GraphConv per (fine, coarse) stack (`configs/model.yaml:18`). | `resblk_num=2` per `GraphResBlocks` at every depth, with the **mid-bottleneck running two extra resblocks** (`graph_diffusion.py:89-93`, config `configs/chinesefont_train.yaml:108`). `resblock_type=basic`. | **MEDIUM** — comparable count, different shape. |
| H3 | **`embed_dim` / `n_embed`** | Not present. | `embed_dim=3`, `n_embed=8192` (`configs/chinesefont_train.yaml:110-111`). These are **vestigial fields inherited from DualOctreeGNN's VQ-VAE codebook variant** — none of the `graph_diffusion.Graph_diffusion` constructor reads them (cf. `graph_diffusion.py:48-50`). | **NIL** — sanity-checked; no parity hit. |
| H4 | **`bottleneck`, `code_channel`** | Not present. | `bottleneck=4`, `code_channel=32` (`configs/chinesefont_train.yaml:105, 107`). `bottleneck` controls only the `bottleneck`-type resblock (we use `basic`, so it's unused); `code_channel` is also unused by `graph_diffusion`. | **NIL** — vestigial. |
| H5 | **Grad clip** | `grad_clip=1.0` (`configs/train_stage_a_ttf.yaml:15`). | Not visible in the snippet we read; check `solver/solver.py` if needed. The training config does not surface a `grad_clip` key. | **LOW** — likely default solver clip; verify before launch. |
| H6 | **DDP / multi-GPU** | Single-GPU assumed. | `gpu: 0,1,2,3,4,5,6,8` (8 GPUs incl. skipping 7) with `dist_url: tcp://localhost:10015` (`configs/chinesefont_train.yaml:9, 13`). Model has `find_unused_parameters: True` (`configs/chinesefont_train.yaml:113`), consistent with `'cond' in graph_diffusion.forward` branching. | **HIGH** — blind impl is single-process; if we ever DDP, set `find_unused_parameters=True` because of the conditional cond branches. |

---

## 6. data_pp_deltas

| # | Topic | Blind impl | Official | Severity |
|---|---|---|---|---|
| D1 | **Source representation** | Pixel image quantised by `adaptive_avg_pool2d → K-way bin` (`model.py:109-133`, `blind_impl.md:72-75`). | **Contour + skeleton point cloud**: glyph → `cv2.findContours` → `drawContours` (1 px) + `skimage.morphology.skeletonize` → 3-class label image `{0=bg, 1=contour, 2=skeleton}` → flattened to points (`datasets/chinesefont_asymmetric.py:303-309, 124-128`). Octree is then built from those points (`build_octree` at `ocnn_2d/octree/octree.py:161`). | **CRITICAL** — paper does NOT take pixel intensities into the diffusion; it takes the **geometric primitives** of the glyph. This is the actual representation pipeline the paper claims. |
| D2 | **Glyph binarisation** | None — diffusion runs over multi-bin pooled intensities. | `(image[:,:,0] > 127.5).astype(uint8)` thresholding before contour extraction (`datasets/chinesefont_asymmetric.py:259`). | **HIGH** — paper assumes binarised input. |
| D3 | **Augmentation** | `RandomAffine + ColorJitter` (cf. peer pipelines). | `distort: False` (`configs/chinesefont_train.yaml:54, 82`). | **MEDIUM** — no augmentation by default. |
| D4 | **Reference selection** | Per-batch random sample from training set. | **Fixed reference list** (`data/VQ-Font128/train_unis.json`), one query char excluded per sample, `ref_num=3` (`datasets/chinesefont_asymmetric.py:231-243, 262-266`). | **MEDIUM** — controlled few-shot regime. |
| D5 | **Cross-font content path** | Content image = source-font render of the same char. | Optional **cross-font content**: when `c_ref_num > 1`, content glyph is drawn from a different font (`datasets/chinesefont_asymmetric.py:276-281`); when `c_ref_num == 1`, content comes from a dedicated `content/0/` directory of canonical renders (`datasets/chinesefont_asymmetric.py:282-283`). | **MEDIUM** — Ernantang has no "canonical font 0" content path; we'll need to define one if porting. |
| D6 | **`canny` flag** | Not present. | `canny: True` toggles the **contour+skeleton extraction described in D1**; if `False`, the pipeline would degrade to a 1-class point cloud (no skeleton supervision). All training configs leave it on. | **HIGH** — feature-flag-controlled but always on; in blind impl this is missing. |
| D7 | **Normalisation** | `[-1, 1]` (`paper_notes/05.md:91-93`). | Points are normalised to `[-1, 1]` via `(p - mid)/mid` (`datasets/chinesefont_asymmetric.py:130`), consistent. | **LOW** — agrees in spirit. |

---

## 7. risk_of_bug

Plausible bugs/gaps in **our blind impl** flagged by the diff (does NOT modify the code — items the reimpl-worker should review):

| # | Symptom we'd see | Suspected root cause | Where to look |
|---|---|---|---|
| R1 | Model converges to a low CE on a near-uniform leaf-state distribution that does not look like text. | We train CE on 8-bin **pixel intensity**, not on geometric labels. There is no signal forcing strokes vs. background; the paper's CE forces structure via 3-class labels `{bg, contour, skeleton}`. | `train.compute_loss` + `quantize_to_states` (`src/qt_font/model.py:109-133`). |
| R2 | Conditioning paths get zero useful gradient even though the per-branch grad smoke check passes. | `cond` is broadcast once at input; in the paper it is re-injected per resblock × per depth (`graph_diffusion.py:148-149`). A single additive injection at depth `d_in` is heavily attenuated by `n_layers` of GraphConv before reaching the output head. | `src/qt_font/model.py` cond-injection point. |
| R3 | Sampler produces blurry / non-axis-aligned glyphs. | (a) Reverse process uses argmax-on-`t=0` rather than full posterior + Gumbel-max sampling (`main.py:258-273`); (b) Only one denoising step per `T_blind=100`, vs. official `gap=50` over `T_official=1000` (still 20 effective steps but with cosine schedule); (c) Output is a continuous pixel image (bilinear upsample of expected bin centres), not a re-rasterised contour+skeleton. | `src/qt_font/sample.py`, decode path in `model.py:140-160`. |
| R4 | `id` embeddings dominate training, model ignores style refs. | Blind impl uses `char_id`, `writer_id`, `script_id` embeddings that the paper does not (cf. C1). With Ernantang's 24 writers × 4659 chars, the embeddings are an easy shortcut and will absorb most of the loss. | `src/qt_font/model.py` id-embedding tables and `cond` sum. |
| R5 | CFG dropout breaks training. | Paper has no CFG; null-token injection on absent paths may interact with the `'cond' in cond_str` branches if we ever copy the official conditional architecture. | `blind_impl.md:104-108`. |
| R6 | Blind impl's `n_states=8` and pixel-intensity bins are off-distribution for binarised renders. | At binarised input, `K=8` will collapse to two modes (0 and 7), wasting 6 classes and making the CE loss nearly degenerate. | `quantize_to_states` + Stage A YAML. |
| R7 | Naming collision: paper "depth" is octree depth (=7 for 128 px) but blind "depth" is full-quadtree depth (=4 for 16×16 leaves). | Blind impl conflates two distinct quantities; any future port of the paper's config will silently produce a 16× shallower model. | `configs/model.yaml:13`. |
| R8 | `axis_loss`'s leaf-level label derivation `F[:, 2]*1 + F[:, 3]*2` (`losses/loss.py:25`) shows that **feature columns 2, 3 of the leaf feature** are the 1-hot contour/skeleton flags. Our `_get_input_feature` analogue does not emit such a feature. | We never wire contour/skeleton point features. | `src/qt_font/model.py` `_get_input_feature` and dataset. |
| R9 | Hyperparams claim "paper-cited wd=1e-2" but the paper config sets `weight_decay: 0.00` (`configs/chinesefont_train.yaml:26`). | The Obsidian note that fed `blind_impl.md:88-90` is wrong. | `blind_impl.md` decision log entry 13. |

---

## 8. Phase 3 compensations (recommended next steps for the reimpl-worker)

The diff is large enough that a **rewrite-to-paper-shape** for QT-Font would essentially mean adopting the official `ocnn_2d` + DualOctreeGNN stack. That is out of scope for blind-impl maintenance. For Phase 3 launch, the **research-defensible compensations** are:

1. **Stage A (TTF) — keep blind impl as a "raster-quadtree D3PM" baseline**, but:
   - Switch `n_states` from 8 → 3 and **derive label maps from `cv2.findContours` + `skimage.morphology.skeletonize`** to align with the paper's 3-class axis target (closes A3, D1, D2, D6, R1, R6, R8).
   - Switch β schedule to **cosine (Glide form)**; bump `T` to 1000; keep `gap=50` reverse-process stride to keep wall-clock comparable (closes S1, S2, S4).
   - Multi-depth CE: emit a logit at every quadtree level and supervise inner levels with the `nempty` (non-empty/split) mask. Requires also returning a parent-level state per sample (closes L1, A7).
   - Drop the `char_id` / `writer_id` / `script_id` embeddings; keep only style refs + content image (closes C1, R4).
   - Drop CFG until parity is established (closes C6, R5).
   - Document explicitly in `paper_notes/05.md` that the blind impl is the **full-saturated** approximation; an `ocnn`-based ragged variant is a Phase 4 follow-up.
2. **Stage B/C** — defer; with Ernantang's writer/script metadata the right move is the simpler `cond='char'` variant of the official model, not the multi-channel content stack blind impl invented.
3. **Hyperparam parity** — set `weight_decay=0.0` (closes S5, R9). Keep batch_size×grad_accum effective batch as large as the GPU allows; do **not** chase 1024 on a 24 GB GPU.
4. **Sampler parity** — implement the `q_posterior_logits + Gumbel-max` reverse process (`main.py:215-273`) on top of our K-way uniform Q. Even with the simpler 4-connectivity graph, this is the single change most likely to fix R3.

---

## 9. Confidence audit (against `blind_impl.md` §Confidence summary)

| Block | Blind confidence | After diff |
|---|---|---|
| D3PM uniform CE loss | high | **revised: medium** — paper uses multi-depth CE on `nempty` masks **and** a 3-class leaf axis CE; blind impl's single CE is structurally incomplete (L1). |
| AdamW / lr / weight decay | high | **revised: medium** — `wd=0.0` in paper, `1e-2` in blind (S5, R9). |
| Quadtree depth = 4 / n_states = 8 / T = 100 | low | **confirmed wrong** — paper depth = 7, K = 3, T = 1000 (A2, A3, S1). |
| Full vs adaptive quadtree | low | **confirmed wrong** — paper is adaptive sparse via `ocnn_2d.build_octree` (A1). |
| Conditioning injection style | low | **confirmed wrong** — paper injects per resblock per depth, no id embeddings (C1, C4). |
| Style / content encoders | low | **confirmed wrong** — paper uses graph U-Nets over points-on-contour octrees, not CNNs (C2, C3). |
| Coarse-graph topology | medium | **revised: low** — "dual" in the paper = two octrees (in / out), not two adjacencies (A5). |

---

## 10. References

- Blind impl source: `papers/05_qt_font/src/qt_font/{model.py, train.py, sample.py, dataset.py, configs/}`.
- Blind notes: `papers/05_qt_font/paper_notes/05.md`; decision log: `papers/05_qt_font/reports/blind_impl.md`.
- Official repo (cloned): `third_party/05_qt_font/`. Key files:
  - `third_party/05_qt_font/configs/chinesefont_train.yaml`
  - `third_party/05_qt_font/models/graph_diffusion.py`
  - `third_party/05_qt_font/models/modules_bn.py`
  - `third_party/05_qt_font/losses/loss.py`
  - `third_party/05_qt_font/main.py` (sampler + training loop)
  - `third_party/05_qt_font/datasets/chinesefont_asymmetric.py`
  - `third_party/05_qt_font/ocnn_2d/octree/octree.py`
- Paper: Liu & Lian, "QT-Font: High-efficiency Font Synthesis via Quadtree-based Diffusion Models", SIGGRAPH 2024, https://dl.acm.org/doi/10.1145/3641519.3657451 (DOI 10.1145/3641519.3657451).
