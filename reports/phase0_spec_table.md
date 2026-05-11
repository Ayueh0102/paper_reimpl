# Phase 0 Spec Table — 8 Papers Reconnaissance

Reconnaissance compiled from Obsidian notes at `/Users/Ayueh/Documents/Obsidian Vault/research/papers/`. Facts cite paper page/section as recorded in the notes. Anything not stated in the notes is marked `[unknown — needs PDF read in Phase 1]`.

Ernantang baseline assumptions: 24 writers / 32 style_families / 84 units / 4659 chars; TTF renders for 13 fonts; **no open IDS dictionary**, **no pre-trained VQGAN codebook**, **no writer trajectory data**, **no physics PDE labels**.

---

## Section 1: Per-paper specification

### 01 — fontdiffuser (FontDiffuser, AAAI 2024)

| Field | Value |
|---|---|
| paper_id | 01 |
| short_name | fontdiffuser |
| venue_year | AAAI 2024 |
| arch_family | U-Net DDPM (pixel-space, noise-to-denoise) [p.1] |
| conditioning | one-shot: 1 source content image + 1 style reference image; MCA fuses multi-scale content, RSI deforms style ref [p.1-2] |
| loss_terms | DDPM denoising loss; Style Contrastive Loss via SCR extractor (same-char-diff-style as negatives) [p.1] |
| noise_schedule | [unknown — needs PDF read in Phase 1] (DDPM, schedule not stated in note) |
| sampler | DDPM [p.1] |
| official_step_count | [unknown — needs PDF read in Phase 1] |
| batch_size | [unknown — needs PDF read in Phase 1] |
| hardware_used | [unknown — needs PDF read in Phase 1] |
| data_needed | (src glyph + ref glyph) image pairs; per-char style negatives for SCR; cross-language (CN→KR) eval set |
| compatible_with_ernantang | **true** — image-to-image one-shot only needs TTF renders + style negatives; Ernantang's 24 writers × per-char pairs satisfy SCR requirement |
| github_url | https://github.com/yeungchenwa/FontDiffuser |
| requires_writer_id | false (style is image-conditioned, not categorical) |
| risk_flags | (a) SCR style extractor pretraining recipe absent; (b) MCA fusion scales unspecified; (c) RSI deformation hyperparams absent; (d) baseline reproduction (DG/MX/CG/CF) is itself a project; (e) cross-language val needs Korean fonts |

### 02 — hfh_font (HFH-Font, SIGGRAPH-Asia 2024)

| Field | Value |
|---|---|
| paper_id | 02 |
| short_name | hfh_font |
| venue_year | SIGGRAPH-Asia 2024 |
| arch_family | Latent Diffusion UNet (64×64 latent via VAE) + Style-guided Super-Resolution to 1024×1024 [p.1-2; note "訓練配置"] |
| conditioning | Component-aware attention conditioning; supports few-shot to mid-shot references (variable count) [p.1] |
| loss_terms | Latent DDPM denoising; SDS distillation loss (multi-step teacher → 1-step student); SR module loss [p.1] |
| noise_schedule | [unknown — needs PDF read in Phase 1] |
| sampler | DDPM 10 steps (multi-step teacher); 1-step SDS student; classifier-free guidance sc=ss=2.0 (note "訓練配置") |
| official_step_count | 10 (teacher), 1 (student post-SDS) |
| batch_size | 64 (small set) / 128 (large set) (note "訓練配置") |
| hardware_used | [unknown — needs PDF read in Phase 1] |
| data_needed | Pre-trained VAE; component-level style annotations; SDS teacher checkpoint before student; SR training pairs (low-res ↔ 1024); reference glyphs in variable counts |
| compatible_with_ernantang | **partially** — Ernantang has writers and chars but **no component-level annotations**; need to either skip component attention or derive components from IDS (which we don't have); 1024×1024 not currently in Ernantang renders |
| github_url | null |
| requires_writer_id | false (style is image-conditioned via components) |
| risk_flags | (a) no public code; (b) component-level supervision missing in Ernantang; (c) 3-stage pipeline (teacher → SDS student → SR); (d) pre-trained VAE checkpoint unspecified; (e) 1024 SR doubles cost; (f) `trailing` timestep detail under-specified |

### 03 — if_font (IF-Font, NeurIPS 2024)

| Field | Value |
|---|---|
| paper_id | 03 |
| short_name | if_font |
| venue_year | NeurIPS 2024 |
| arch_family | VQGAN tokenizer + Autoregressive Transformer decoder (10 blocks, 8 heads, dim=384, 2 self-attn + 1 cross-attn per block) [p.1; note "訓練配置"] |
| conditioning | **IDS (Ideographic Description Sequence)** text via text encoder + reference glyph VQ tokens via VQGAN encoder; source-glyph-free [p.1] |
| loss_terms | AR cross-entropy on target VQ tokens; VQGAN reconstruction (pre-training stage) [p.1] |
| noise_schedule | n/a (AR, not diffusion) |
| sampler | autoregressive token sampling |
| official_step_count | n/a |
| batch_size | 128 (note "訓練配置") |
| hardware_used | 1× NVIDIA V100 16GB, 15 epochs, ~42h (note "訓練配置") |
| data_needed | **IDS dictionary** (e.g. CHISE) covering target chars; pre-trained VQGAN font codebook (size 256, downsample 8); reference glyph images |
| compatible_with_ernantang | **partially** — IDS dictionary for Ernantang's 4659 chars is obtainable from CHISE/Unicode IDS but is an external dependency we don't currently have plumbed; VQGAN codebook must be trained on Ernantang/TTF before main training |
| github_url | https://github.com/Stareven233/IF-Font |
| requires_writer_id | false (style via reference image VQ tokens) |
| risk_flags | (a) external IDS dictionary required (CHISE/Unicode IDS); (b) VQGAN pre-training is an upstream sub-experiment; (c) AR scan order unspecified; (d) IDS text encoder choice unspecified; (e) codebook size 256 may be tight for 4659 chars |

### 04 — vq_font (VQ-Font, AAAI 2023)

| Field | Value |
|---|---|
| paper_id | 04 |
| short_name | vq_font |
| venue_year | AAAI 2023 |
| arch_family | Two-stage: any FFG synthesis module → Transformer-based Token Prior Refinement over pre-trained VQGAN codebook (codebook size 1024, 16×16 features) [p.1; note "訓練配置"] |
| conditioning | Source char glyph + 3 reference glyphs; SSEM uses **12 Chinese structure categories** (左右/上下/包圍 etc) for structure-level style matching [p.1] |
| loss_terms | VQGAN reconstruction (pre-training); cross-entropy on codebook index prediction; SSEM structure-level style loss [p.1] |
| noise_schedule | n/a |
| sampler | argmax/sample over codebook indices |
| official_step_count | n/a |
| batch_size | 32 (note "訓練配置") |
| hardware_used | 1× A6000; VQGAN pre-train 200k iters + token refinement 300k iters (note "訓練配置") |
| data_needed | Pre-trained VQGAN font codebook (must train); 12-structure classification label per char; 3 reference glyphs per query; synthesis-module output as input |
| compatible_with_ernantang | **partially** — Ernantang has writers/chars but no 12-structure labels (derivable from IDS / Unicode CJK structural property); needs upstream synthesis module + VQGAN pretrain |
| github_url | https://github.com/Yaomingshuai/VQ-Font |
| requires_writer_id | false (style image-based) |
| risk_flags | (a) 12-structure labels missing in Ernantang; (b) VQGAN codebook pre-training is upstream; (c) needs working synthesis module as prerequisite; (d) token-prior-refinement Transformer details under-specified; (e) assumes 3-reference setup |

### 05 — qt_font (QT-Font, SIGGRAPH 2024)

| Field | Value |
|---|---|
| paper_id | 05 |
| short_name | qt_font |
| venue_year | SIGGRAPH 2024 |
| arch_family | Dual Quadtree Graph U-Net + Discrete Diffusion in quadtree node space; content-aware pooling [p.1-2] |
| conditioning | Reference glyphs (quadtree-encoded) for few-shot setup [p.1] |
| loss_terms | Discrete diffusion cross-entropy on quadtree node states; content-aware pooling regularizer (not detailed in note) [p.1] |
| noise_schedule | discrete diffusion (categorical), schedule [unknown — needs PDF read in Phase 1] |
| sampler | discrete diffusion reverse process |
| official_step_count | [unknown — needs PDF read in Phase 1] |
| batch_size | pretrain batch=1024 (gradient accumulation) / fine-tune batch=8 (note "訓練配置") |
| hardware_used | [unknown — needs PDF read in Phase 1]; AdamW β1=0.9 β2=0.999 wd=0.01 lr=1e-4 cosine; pretrain 20 epochs (~16k steps) |
| data_needed | Rasterized outline → sparse point cloud → quadtree conversion pipeline; reference glyphs; high-res (256/512) glyph targets |
| compatible_with_ernantang | **partially** — Ernantang TTF renders allow outline extraction and high-res rasterization, but the quadtree representation pipeline and dual-graph-U-Net are non-trivial implementations not in standard libs |
| github_url | null |
| requires_writer_id | false (style via reference images) |
| risk_flags | (a) no public code; (b) quadtree construction pipeline from scratch; (c) discrete diffusion on graphs has no off-the-shelf lib; (d) content-aware pooling under-specified; (e) effective batch 1024 via accumulation is heavy; (f) hardest from-scratch impl in the set |

### 06 — calliffusion (Calliffusion, AAAI 2024 workshop)

| Field | Value |
|---|---|
| paper_id | 06 |
| short_name | calliffusion |
| venue_year | arXiv 2023 / AAAI 2024 workshop (note: vault file `_ISLAND_031` is mismatched PDF; correct note is `Liao_2023_arXiv_Calliffusion_...`) |
| arch_family | Conditional DDPM U-Net (blocks dims [320, 640, 1280, 1280]) [p.2-3] |
| conditioning | Chinese BERT text embedding (dim=768) of "char + script + calligrapher style" prompt, via cross-attention at each resolution [p.2] |
| loss_terms | DDPM denoising loss; LoRA fine-tuning loss for one-shot style transfer [p.2] |
| noise_schedule | linear variance schedule β₁→βN (note "訓練配置") |
| sampler | DDPM |
| official_step_count | [unknown — needs PDF read in Phase 1] |
| batch_size | 16 (note "訓練配置") |
| hardware_used | 2× A100 40GB × 120h (note "訓練配置") |
| data_needed | Calligraphy images with (char + script + calligrapher) text labels; pre-trained Chinese BERT; LoRA fine-tune samples for style transfer |
| compatible_with_ernantang | **partially** — Ernantang has writer/script/char metadata; can build "<char> <script> <writer>" prompts; needs Chinese BERT (公開可下載). Style transfer via LoRA is straightforward. Ernantang has 24 writers; paper used 1387 — pre-training scale is much smaller for us. |
| github_url | null |
| requires_writer_id | indirect — calligrapher name is a BERT token, not a categorical id |
| risk_flags | (a) no public code; (b) BERT weak on calligrapher names (per Moyun paper); (c) 2×A100 × 120h is heavy for single-GPU repro; (d) base dataset web-scraped, not directly replicable; (e) vault PDF is the wrong paper — requires re-download |

### 07 — moyun (Moyun, ACM McGE Workshop 2025 / arXiv 2024)

| Field | Value |
|---|---|
| paper_id | 07 |
| short_name | moyun |
| venue_year | ACM McGE Workshop 2025 (arXiv 2024) |
| arch_family | Latent Diffusion (VAE latent 32×32×4) with **Vision Mamba (Mamba2)** replacing U-Net; DiT-style scale-shift modulation [p.3-4] |
| conditioning | **TripleLabel**: three independent trainable embeddings (calligrapher_id + font/script_id + char_id), summed → MLP+SiLU → scale-shift on Mamba blocks; classifier-free guidance [p.4] |
| loss_terms | Latent DDPM denoising loss; CFG (no separate aux loss noted) [p.4] |
| noise_schedule | [unknown — needs PDF read in Phase 1] |
| sampler | DDPM (latent) [p.4] |
| official_step_count | [unknown — needs PDF read in Phase 1] |
| batch_size | global batch 768 across 3 GPUs (note "訓練配置") |
| hardware_used | 3× A100; 288,000 iterations; lr 1e-4 (note "訓練配置") |
| data_needed | Pre-trained VAE; **Mobao dataset** (1.93M images, 2681 calligraphers, 4660 chars, 6 scripts) for full repro OR Ernantang as smaller replacement; SAM + k-means binarization pipeline for new data |
| compatible_with_ernantang | **partially** — TripleLabel maps directly to Ernantang's (writer_id, script_id, char_id), excellent fit. But Vision Mamba is a heavier infrastructure dependency (mamba-ssm CUDA package). Ernantang's 24 calligraphers ≪ Mobao's 2681; full-scale repro on Mobao unrealistic without dataset access (Mobao public release status unclear). |
| github_url | null |
| requires_writer_id | **true** (categorical calligrapher embedding) |
| risk_flags | (a) no public code; (b) Mobao dataset access/license unclear; (c) mamba-ssm CUDA toolchain dependency; (d) pretrained VAE unspecified; (e) 3×A100 × 288k iters heavy for single-GPU repro; (f) cursive/clerical/seal OCR low even at SOTA — script-stratified eval needed |

### 08 — dp_font (DP-Font, IJCAI 2024)

| Field | Value |
|---|---|
| paper_id | 08 |
| short_name | dp_font |
| venue_year | IJCAI 2024 |
| arch_family | DDPM U-Net + **PINN (Physics-Informed NN)** loss [p.1] |
| conditioning | Multi-attribute guidance (writer ID, ink intensity, font size, etc.) + **stroke order** sequence as fine-grained constraint; classifier-free guidance (ω∈[0,1]) [p.1; note "訓練配置"] |
| loss_terms | L_simple (DDPM denoising) + L_PINN (physics PDE residual MSE for nib motion + ink diffusion) [p.1] |
| noise_schedule | [unknown — needs PDF read in Phase 1] |
| sampler | DDPM with CFG |
| official_step_count | [unknown — needs PDF read in Phase 1] |
| batch_size | [unknown — needs PDF read in Phase 1] |
| hardware_used | single GeForce RTX 3090; input 80×80 (note "訓練配置") |
| data_needed | Liu Gongquan / Yan Zhenqing calligraphy renders (80×80); **stroke order sequence per char**; **ink diffusion / nib motion PDE labels or analytical form**; multi-attribute labels (writer, ink, size) |
| compatible_with_ernantang | **partially → false at full scale** — Ernantang has writer/char but **no stroke-order data**, **no ink/nib physics ground truth**. Stroke-order is derivable from public stroke-order DB for common chars, but our 4659 chars may not all have entries. PINN PDE form not given in note → may require domain modeling from scratch. |
| github_url | null |
| requires_writer_id | true (writer is one of the multi-attribute conditions) |
| risk_flags | (a) no public code; (b) PINN PDE form not in note — likely partial re-derivation; (c) stroke-order coverage gap for 4659 chars; (d) ink-diffusion GT never measured in Ernantang; (e) "physical plausibility" eval metric paper-defined, ambiguous to re-impl; (f) 80×80 res mismatches Ernantang's higher-res preference |

---

## Section 2: Cross-paper risk register

### 2.1 Shared dependencies blocking multiple papers

| Dependency | Affects | Why it's a blocker |
|---|---|---|
| **External IDS dictionary** (CHISE / Unicode IDS) | 03 if_font (hard requirement), 04 vq_font (12-structure derivation), 08 dp_font (stroke-order is adjacent) | Ernantang ships no IDS; obtaining and aligning IDS to 4659 chars is a Phase 1 prereq, not free work |
| **Pre-trained VQGAN font codebook** | 03 if_font, 04 vq_font | Both papers train their own codebook on font corpora before main model; ~200k iters of VQGAN on font data is itself a multi-day job. Codebook sizes differ (256 vs 1024) so can't be shared. |
| **Pre-trained VAE (image → latent)** | 02 hfh_font, 07 moyun | Both assume a VAE that compresses glyph image to small latent. SD-1.5 VAE is one default but not stated. Quality on binary glyphs is suspect — may need finetune. |
| **No public code (github = null)** | 02 hfh_font, 05 qt_font, 06 calliffusion, 07 moyun, 08 dp_font (5 of 8) | Five papers have `facts_code_url: null`. Blind re-implementation risk is substantially higher for these five. |
| **Writer-id as categorical input** | 07 moyun, 08 dp_font | Ernantang's 24 writers is fine for embedding tables, but evaluation against papers (Moyun used 2681) means our results are not directly comparable. |
| **Stroke-order sequence per char** | 08 dp_font (hard), 04 vq_font (soft, via structure) | Ernantang has no stroke-order. Public DB (e.g. cjklib / Make Me a Hanzi) covers many but not all chars. Coverage audit required before launch. |

### 2.2 Compute budget vs single RTX 6000 Ada 48GB

- **06 calliffusion**: 2×A100 × 120h → marginal on single GPU; subset repro recommended.
- **07 moyun**: 3×A100 × 288k iters @ batch 768 → ~3× wall-clock single-GPU; subset repro (Ernantang scale) recommended.
- **03 if_font**: 1×V100 × 42h batch 128 → fully feasible.
- **04 vq_font**: 1×A6000 × (200k + 300k iters) → fully feasible.
- **08 dp_font**: single 3090, 80×80 → fully feasible (smallest budget).
- **01 fontdiffuser, 02 hfh_font, 05 qt_font**: hardware not stated; assume A100-class. PDF audit required before launch.

### 2.3 Data gaps vs Ernantang

| Paper | Ernantang gap | Mitigation |
|---|---|---|
| 01 fontdiffuser | none significant | direct training |
| 02 hfh_font | component-level annotations; 1024×1024 renders | derive components from IDS (after 03 prereq done) or skip the component module |
| 03 if_font | IDS dictionary | external (CHISE) — one-time ingest |
| 04 vq_font | 12-structure classification labels | derive from Unicode IDS structural properties |
| 05 qt_font | quadtree pipeline | implement from scratch; outline data fine |
| 06 calliffusion | scale (Ernantang << paper's 1387 calligraphers); writer-name BERT tokens | embed writer names as Chinese strings via BERT; live with smaller scale |
| 07 moyun | Mobao dataset access; calligrapher count (24 vs 2681) | swap Mobao → Ernantang; report scale-down |
| 08 dp_font | stroke-order labels; PINN PDE ground truth | external stroke-order DB; PINN may need to be dropped or reformulated |

---

## Section 3: Suggested Phase 1 priority order

1. **01 fontdiffuser** — pipeline shakedown. github exists; image-in/image-out is the simplest contract with Ernantang; no external data dependency; DDPM U-Net is the most boilerplate arch in the set. Original recommendation stands.
2. **03 if_font** — github exists; prereq is IDS dictionary ingestion (a self-contained sub-task that also unlocks 04 + partial 02). VQGAN pre-training is upstream but the Transformer module can be validated with a random codebook stub first.
3. **04 vq_font** — github exists; reuses IDS plumbing from step 2 for 12-structure labels. Two-stage design lets us validate token-prior-refinement using a frozen FontDiffuser (step 1) as the synthesis module — gives Phase 2 a useful comparison.
4. **08 dp_font** — smallest compute footprint (single 3090, 80×80). Resolve stroke-order data coverage before launch; PINN PDE may be downgraded to placeholder if note-level info is insufficient.
5. **06 calliffusion** — straightforward U-Net DDPM + Chinese BERT, but no github and the vault PDF is wrong (re-download required). Scale-down to Ernantang is expected.
6. **02 hfh_font** — no github + 3-stage pipeline (teacher → SDS distill → SR). Defer until at least one latent-diffusion entry has been stood up so VAE plumbing is shared.
7. **07 moyun** — Vision Mamba is a hard infra dependency (mamba-ssm CUDA wheel). TripleLabel maps cleanly to Ernantang metadata but the arch swap is the riskiest part.
8. **05 qt_font** — riskiest end-to-end: no github, niche representation (quadtree + graph + discrete diffusion), no off-the-shelf library. Treat as research-grade — may not converge to paper quality on first pass.

Tie-breakers: (i) low-risk + has-github before high-risk + no-github; (ii) papers producing shared infra (VAE, IDS, VQGAN codebook) before consumers; (iii) compute feasibility on single 6000 Ada.

---

End of Phase 0 spec table.
