# Calliffusion / CalliffusionV2 — GitHub Diff (Phase 2)

**Status: `official_unavailable`**

## 1. Repo search summary

No official source-code release was located for either:

- **Calliffusion v1** — Liao, Xia, Wang, arXiv:2305.19124 (2023-05-30)
- **CalliffusionV2**  — Liao, Li, Fei, Xia, arXiv:2410.03787 (2024-10)

Sources checked:

| # | Source | Result |
|---|--------|--------|
| 1 | `arxiv.org/abs/2305.19124` abstract page | No GitHub / project URL in metadata or comments |
| 2 | `arxiv.org/abs/2410.03787` (V2) abstract page | No GitHub / project URL in metadata or comments |
| 3 | `ar5iv.labs.arxiv.org/html/2305.19124` (full HTML) | Searched all sections, references, footnotes — no `github.com`, no "code available at", no project page URL |
| 4 | `ar5iv.labs.arxiv.org/html/2410.03787` (V2 full HTML) | Same — no code-release statement |
| 5 | `huggingface.co/papers/2305.19124` (auto-aggregated PwC entry) | "Request Code" placeholder, no link present |
| 6 | `catalyzex.com/author/Qisheng%20Liao` | Three calligraphy papers (Calliffusion, CalliPaint, CalliffusionV2) all show "Request Code"; only unrelated `QishengL/SemEval2023` repo (framing detection) is listed |
| 7 | `github.com/QishengL` (first author's GitHub) | Only `diffusers_hackedsd3` (SD3 encoder hack, unrelated) and `wav2vec_test`. No calligraphy repo |
| 8 | GitHub global search `"calliffusion"` | Zero matching repos. Hits are unrelated (`CalliECD`, `Awesome-LoRAs` listings, `causalfusion`, `awesome-coldfusion`) |
| 9 | Hugging Face Spaces search `calliffusion` | "No result found" |
| 10 | `paperswithcode.com/paper/calliffusion-...` | Redirects to HF papers page — no code attached |

**Conclusion:** As of 2026-05-11 the authors have not released training code, inference code, weights, or a project website. Reproducing Calliffusion must therefore be driven by the paper text alone. No source diff is possible; the rest of this report compares our Phase-2 guesses against the *paper's prose*, which is the closest substitute.

## 2. Paper-vs-our-guess diff (no code; quotes from ar5iv HTML)

Quotes are verbatim from `https://ar5iv.labs.arxiv.org/html/2305.19124` (HTML mirror of v1; v2 paper adds multimodal control but is silent on most low-level hyperparameters).

### 2.1 U-Net depth/widths — `[320, 640, 1280, 1280]`

- **Our Phase-2 guess:** `[320, 640, 1280, 1280]`, four down-blocks.
- **Paper, §Training → Hyperparameters:** "We configured four blocks in the U-Net architecture with dimensions of 320, 640, 1280, and 1280, each consisting of two layers."
- **Diff:** **MATCH.** Our guess is exact. Note the paper also specifies "two layers" per block, i.e. `layers_per_block=2` in `diffusers`-style configs.

### 2.2 BERT checkpoint

- **Our Phase-2 guess:** `bert-base-chinese`.
- **Paper, §Adding Controls with External Conditions:** "The input text is then passed through a pre-trained Chinese BERT model to obtain cross-attention embeddings with a size of 768"
- **Diff:** **PARTIAL.** Embedding dim 768 is consistent with `bert-base-chinese` (12-layer, 768-hidden), but the paper does not name the checkpoint. `chinese-bert-wwm`, `chinese-roberta-wwm-ext`, `MacBERT-base-chinese`, and `hfl/chinese-bert-wwm-ext` all also have 768-dim outputs. **Assumption needs to stay flagged**; pick `bert-base-chinese` as the safest default but log the choice in the eval card.

### 2.3 LoRA placement / rank

- **Our Phase-2 guess:** cross-attention only.
- **Paper, §Style Transfer:** "LoRA achieves this by adding update matrices, which are rank-decomposed weight matrices, to the existing weights" — and (V2): "We add two additional trainable matrices into the U-Net".
- **Diff:** **UNDERSPECIFIED.** Neither paper states which modules are wrapped (cross-attn only vs. all Linear vs. attn QKV). No rank `r`, no alpha, no dropout. The phrase "two additional trainable matrices" in V2 weakly suggests a single linear adapter pair per block but is ambiguous. **Default to the `diffusers`/PEFT calligraphy convention: rank=4 on cross-attn QKV+out only**, and treat this as our own design choice rather than a faithful reproduction.

### 2.4 CFG dropout `p`

- **Our Phase-2 guess:** `p_drop = 0.1`.
- **Paper:** No mention of classifier-free guidance dropout in either v1 or v2. CFG itself is not explicitly discussed in v1's training section.
- **Diff:** **NOT IN PAPER.** Our 0.1 is the SD-1.x default and is a reasonable inference of intent, but cannot be cited as Calliffusion-faithful. Document as "borrowed from Rombach et al. 2022 / SD defaults".

### 2.5 Special-token registration for artist names

- **Our Phase-2 guess:** dedicated learned tokens per artist registered into BERT's vocab.
- **Paper, §Adding Controls with External Conditions:** "We use a short description of Chinese text input, such as '人字 隶书 曹全碑' … The text consists of three parts, and a space separates each part. The first part of the text determines the character, the second part controls the script, and the last part determines the calligrapher's style."
- **Diff:** **MISMATCH (likely).** Calliffusion appears to feed artist names as **plain CJK strings** through the off-the-shelf Chinese BERT tokenizer, relying on BERT's existing char-level tokenization (artist names like 曹全碑, 颜真卿 split into normal Han chars). No evidence of `add_special_tokens` / new embedding rows. Our "register one token per artist" plan is a *stronger* design but is not the paper's mechanism. Either:
  - **(a)** Drop the special-token plan and reproduce the plain-prompt approach (faithful, simpler, and BERT already covers the character distribution); or
  - **(b)** Keep the special-token plan as an explicit *extension* and label it in the ablation table.

### 2.6 Other concrete numbers found while diffing (use as known facts, not guesses)

| Item | Paper quote | Section |
|------|-------------|---------|
| Optimizer / LR | "Adam optimizer with a learning rate of 1×10⁻⁵" | Hyperparameters |
| Batch size | "batch size was set to 16" | Hyperparameters |
| Training compute | "two NVIDIA A100 40G GPUs for a total of 120 hours" | Hyperparameters |
| Layers per U-Net block | "each consisting of two layers" | Hyperparameters |
| BERT output dim | "embeddings with a size of 768" | Adding Controls |
| Dataset (raw) | "5 scripts, … 3975 unique characters and 1431 artists" | Dataset |
| Dataset (filtered) | "reduced dataset of 2025 characters and 1387 artists" | Dataset |
| Prompt format | `"人字 隶书 曹全碑"` (char, script, artist) | Adding Controls |
| Image size | Not specified in v1 HTML | — |
| Number of timesteps / schedule | Not specified in extracted text | — |
| LoRA rank/alpha/dropout | Not specified | — |
| CFG p | Not specified | — |
| VAE / latent vs pixel space | Not specified in v1 HTML (architecture suggests pixel-space DDPM, but unconfirmed) | — |

## 3. Recommendations for Phase 3

1. **Treat all hyperparameter guesses as our own choices**, not as "Calliffusion-faithful". Add an explicit `assumptions.md` for paper 06 listing every value where the paper is silent.
2. **Keep U-Net widths `[320, 640, 1280, 1280]`, `layers_per_block=2`, optimizer Adam @ 1e-5, batch 16** — these are exact-match values from the paper, so they are safe anchors.
3. **For BERT, default to `bert-base-chinese`** but record the decision; revisit if we ever get author confirmation.
4. **For artist conditioning, run BOTH variants** in our ablation: (a) plain-prompt char/script/artist (paper-faithful), (b) registered special token (our extension). This directly satisfies CLAUDE.md rule #4 ("every claim needs an ablation").
5. **Mark LoRA rank, CFG `p_drop`, image resolution, diffusion schedule as `paper_silent`** in `06_calliffusion/notes.md`. Pick defaults from SD 1.x but do not advertise as reproductions.
6. **Stop searching for a Calliffusion repo.** Re-check once per quarter only — author has had 24+ months to release, no signal of intent. CalliffusionV2 demo URL is also not present in the paper.

## 4. Sources

- arXiv abstract v1 — https://arxiv.org/abs/2305.19124
- arXiv abstract V2 — https://arxiv.org/abs/2410.03787
- ar5iv HTML v1 (primary source for quotes) — https://ar5iv.labs.arxiv.org/html/2305.19124
- ar5iv HTML V2 — https://ar5iv.labs.arxiv.org/html/2410.03787
- catalyzeX author page — https://www.catalyzex.com/author/Qisheng%20Liao
- First-author GitHub — https://github.com/QishengL
- Hugging Face papers — https://huggingface.co/papers/2305.19124
