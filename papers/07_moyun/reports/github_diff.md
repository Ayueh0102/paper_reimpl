# Phase 2 GitHub Diff — Paper 07: Moyun (McGE '25 / arXiv 2410.07618)

## Status: `official_unavailable`

Moyun does not publish an official code repository. Phase 0 left the GitHub URL blank, and Phase 2 search confirms there is no public release as of 2026-05-11.

### Search trail

1. WebSearch — "Moyun diffusion mamba calligraphy github code release 2024 2025": no official repo, only Awesome lists and unrelated namespace collisions.
2. WebSearch — `"Moyun" calligraphy "TripleLabel" Mamba site:github.com`: surfaces `github.com/aixxx/moyun`, verified to be a PHP FastAdmin scaffold (false positive on the name string).
3. WebSearch — `"Moyun" "Mobao" dataset calligraphy github source code`: surfaces `github.com/moyunmo`, verified to be an iOS developer (CMTabbarView, ReactiveCocoa) — false positive on username.
4. WebSearch — `"Kaiyuan Liu" "Jiahao Mei" Moyun Mamba github`: first author Kaiyuan Liu's homepage (`lkytal.github.io`) and `github.com/56dot1` checked — no Moyun repo; their public repos are about MLsys, on-device LLM inference, classical-Chinese NMT.
5. paperswithcode (`/paper/moyun-a-diffusion-based-model-for-style`): 302-redirects to `huggingface.co/papers/2410.07618`; no code link on either page.
6. arXiv abstract (https://arxiv.org/abs/2410.07618) and v2 HTML (https://arxiv.org/html/2410.07618v2): no `code available at` / `https://github.com/` statement; PDF body confirms the same.
7. Awesome-Vision-Mamba-Models raw README (`Ruixxxx/Awesome-Vision-Mamba-Models`): Moyun row is listed (`Arxiv 24.10.10 | Moyun: A Diffusion-Based Model...`) but the Code column is empty (`| [Link](arxiv) | |`).
8. GitHub API (`/search/repositories?q=moyun+calligraphy+mamba`, `q=moyun+diffusion`, `q=mobao+calligraphy`): all return `total_count: 0`.

No clone, no diff. The rest of this document compares the paper's described design (v2 HTML, Sections 3.2–3.3, 4.2) against our current implementation, since that is the actionable substitute for a code diff.

---

## Paper-described design vs our codebase

### 1. Mamba block (pure-PyTorch S6 vs paper claim)

- Paper (arxiv 2410.07618v2 §3.2): "Vision Mamba … while incorporating the advanced Mamba2." No library citation; no bidirectional scan rule (concat-and-project / flip-add / average) given; no kernel-level details.
- Ours: no Mamba at all. Backbone is a DiT-style Transformer with AdaLN-Zero modulation. Confirmed by:
  - `src/ernantang_jit/model.py:44` `AdaLNZeroBlock` (self-attn + MLP, 6-chunk AdaLN-Zero).
  - `src/ernantang_jit/model.py:471` `UnitAwareJiT` docstring: "All-Transformer diffusion backbone with AdaLN-zero conditioning."
  - `src/ernantang_jit/model.py:636-647` `_make_block` only emits Transformer variants (`AdaLNZeroBlock`, `ContentCrossAdaLNZeroBlock`, `RefCrossAdaLNZeroBlock`, `MultiCrossAdaLNZeroBlock`); no Mamba/SSM branch.
- Diff: not directly comparable. Without their code we cannot tell whether they use `mamba-ssm`/`mamba2-ssm` (CUDA kernel), a pure-PyTorch S6 reference, or a third-party Vision-Mamba variant (Vim, VMamba). For our research-first plan this is not a blocker; we stay Transformer and only borrow their conditioning shape.

### 2. TripleLabel embedding (separate-then-sum)

- Paper (§3.3): "each label is mapped to a unique class label … through three separate trainable embedding tables"; combined as `e_total = e_calli + e_font + e_char`; enables "zero-shot generalization to unseen combinations through linear superposition."
- Ours: same separate-tables-then-sum pattern, but extended to four axes (writer, style_family/unit, char, script):
  - `src/ernantang_jit/model.py:586-593` declares `writer_emb`, `style_family_emb`, `char_emb`, `script_emb` as independent `nn.Embedding` tables sized to per-axis vocab.
  - `src/ernantang_jit/model.py:705-712` `_condition` sums them onto the time-MLP base: `cond = self._add_axis_cond(cond, "writer", …)` then style_family, char, script — i.e. `cond = t + e_writer + e_style + e_char + e_script (+ ref_pool)`.
- Diff: structurally identical embedding-superposition. Moyun: 3 axes. Ours: 4 axes (we keep `script` as a separate trainable axis instead of folding it into `font`). We also have a learned `[NULL]` per axis for CFG (`model.py:597-604` `cond_null_embs`), which the paper does not describe.

### 3. Bidirectional scan mechanism

- Paper: not specified in v2 (confirmed by WebFetch over both PDF and HTML). The bidirectional-scan rule (concat-and-project vs flip-add vs average) is exactly the implementation detail that is missing from the paper and would require their code to resolve.
- Ours: N/A — no SSM scan; self-attention is order-equivariant after `pos_embed` is added.
- Diff: unresolvable. If we ever port their backbone, we will have to pick a Vim-style concat-and-project or VMamba-style 2D bidirectional cross-scan as a default and ablate against it; the paper does not constrain the choice.

### 4. AdaLN-Zero modulation specifics

- Paper (§3.3): "MLP with SiLU activation" produces modulation parameters "{α, γ, β}" used in a "scale-shift mechanism" (DiT reference). No gate-zero-init statement, no order-of-application detail.
- Ours: full DiT AdaLN-Zero contract, with explicit zero-init on the modulation projection and the final head.
  - Affine modulator: `model.py:51-53` `self.ada = Sequential(SiLU, Linear(d, d*6))` with `nn.init.zeros_(self.ada[-1].weight/bias)` (zero-init).
  - Forward: `model.py:56-62` chunks into `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`; applies `modulate` (line 27-28: `x*(1+scale)+shift`) then `x = x + gate * residual`. So our chunk count is 6 (self-attn + MLP), the cross-attn blocks extend this to 9 / 12 chunks (`model.py:75, 333`).
  - Final head: `model.py:649-654` `final_ada` (2-chunk) plus zero-init on `head.weight/bias`.
- Diff: aligned in spirit; ours specifies the gate-zero-init that the paper leaves implicit. The paper's "{α, γ, β}" likely corresponds to our `(gate, scale, shift)` triple per residual branch.

### 5. VAE / latent space

- Paper (§4.2): "we employed the same pre-trained VAE as used in LDM … 256×256×3 → 32×32×4." No checkpoint URL, no whether they fine-tune VAE.
- Ours: pixel-space, single-channel grayscale. No VAE encoder.
  - `model.py:567-573` `patchify` is a direct `Conv2d(1, d_model, patch_size, stride=patch_size)` over the noisy image, with optional channel-concat against the content conditioning image (`model.py:572 in_channels = 1 + content_channels`).
  - `model.py:715-723` `unpatchify` returns shape `[B, 1, side*p, side*p]` — grayscale pixel output.
- Diff: meaningful. They run a 256×256 RGB diffusion in 32×32×4 latent; we run a 256×256 grayscale diffusion in pixel space at patch granularity (default `patch_size=16` → 16×16 token grid). If we later want their 32×32 token economy we would need either a smaller `patch_size` (8) or an actual VAE encoder/decoder.

### 6. Training schedule (3×A100, 288k steps, batch 768)

- Paper (§4.2 / our notes): 3× A100, global batch size 768, 288,000 steps (~19,199 epochs), learning rate 1e-4 fixed. Optimizer and CFG dropout rate not stated.
- Ours: see `experiments/A_unit_geometry_jit/` configs (out of scope to enumerate per yaml here). The point of comparison is the order-of-magnitude scale gap — Moyun consumed ~2.2×10⁸ image-steps on a 1.9M-image dataset, which is well beyond a research-stage run on our 4k-unit subset.
- Diff: not architectural. Noted only so we do not blindly try to match 288k steps × bs 768 on a single RTX 5080 16GB — that would take ~weeks for no scientific gain on our writer/unit-conditioned subset.

---

## Action items that survive the missing code

1. Do not block on porting Mamba. Our Transformer + AdaLN-Zero is already the conditioning pattern they imitate.
2. Keep separate-table-sum for writer / style_family / char / script — we already do this and it matches the paper's TripleLabel philosophy.
3. If we ever want to ablate "TripleLabel vs shared", the codebase already supports it via the per-axis null embedding and per-axis drop masks (`model.py:597-604`, `model.py:702-710`).
4. The bidirectional Mamba scan detail is the one piece we cannot recover without their code. Park it; revisit only if we decide to actually swap the backbone.
5. Do not adopt their VAE/latent shape without an explicit experiment plan. Grayscale pixel space matches our binarized data ("Mobao" itself is binarized).
