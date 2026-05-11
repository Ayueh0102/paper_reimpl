# GitHub Diff — 02_hfh_font

**STATUS**: official_unavailable
**Date checked**: 2026-05-11
**Paper**: Li & Lian, "HFH-Font: Few-shot Chinese Font Synthesis with Higher Quality, Faster Speed, and Higher Resolution", SIGGRAPH Asia 2024 (ACM TOG 43(6), Dec 2024)

## Search attempts

Sources checked:

- arXiv abstract page: https://arxiv.org/abs/2410.06488
  - Comments field cites code at `https://github.com/grovessss/HFH-Font` (this is the only code URL authors point to).
- arXiv HTML v1: https://arxiv.org/html/2410.06488v1 — no additional URL.
- ACM Digital Library: https://dl.acm.org/doi/10.1145/3687994 — returned HTTP 403 to WebFetch; metadata via search shows no alternative code link.
- Semantic Scholar / aimodels.fyi / linnk.ai / ResearchGate / SIGGRAPH Asia 2024 papers list (realtimerendering.com/kesen/siga2024Papers.htm) — all point to the same `grovessss/HFH-Font` repo, no mirror.
- Supplemental material Google Drive (`https://drive.google.com/file/d/1CNjsLuqUBskdDwpwtO_affxKx1-MMyQj/view`) — linked from README; supplemental PDF/figures only, no source.
- Author web search (`grovessss` + Zhouhui Lian + Peking University) — no alternate org/user repo found. Author's group at Peking University (Wangxuan Institute) has other public repos (e.g. `yizhiwang96/deepvecfont_homepage`) but none is HFH-Font.

Search queries used:

- `HFH-Font github few-shot Chinese font SIGGRAPH Asia 2024`
- `"HFH-Font" code repository official implementation`
- `"grovessss" HFH-Font Peking University Zhouhui Lian font diffusion github`

## Repository state at https://github.com/grovessss/HFH-Font

- HTTP 200, repo public.
- Single branch `main` at commit `e942919` ("Update README.md").
- Releases page: "There aren't any releases here".
- Tags: none.
- Files: only `README.md` containing:

  > paper: arxiv | supplemental material: here
  > code & data will be available soon!

No source code, configs, requirements, weights, or data pipeline are present. Cloned (depth 1) at 2026-05-11 06:07 UTC for verification; removed since there is nothing to diff against `papers/02_hfh_font/src/`.

## Conclusion

No public official implementation of HFH-Font exists as of 2026-05-11. The author's placeholder repo has been live for 19+ months since arXiv preprint (Oct 2024) and 17+ months since SIGGRAPH Asia camera-ready (Dec 2024) without a code drop. Treat as `official_unavailable` for the foreseeable future.

## Compensating measures

Phase 3 training will rely on:

- Strict adherence to paper sections cited in `paper_notes/02.md` (component-aware conditional LDM, SDS 1-step distillation, style-guided cascaded super-resolution).
- DL reviewer rigor at Stage A/B/C gates (see `reports/dl_review_blind.md`).
- Our [guessed-] decisions in `reports/blind_impl.md` flagged for ablation, with extra caution for:
  - SDS distillation hyperparameters (CFG weight, teacher-student schedule) — paper-only, no reference impl.
  - Component encoder architecture / radical decomposition source — paper-only.
  - Super-resolution cascade resolution ladder and conditioning — paper-only.
- Cross-checking against publicly available diffusion-font baselines we have cloned: `third_party/01_fontdiffuser/`, `third_party/03_if_font/`, `third_party/04_vq_font/` — useful for shared infra (latent diffusion scaffolding, glyph data loaders) but **not** for HFH-Font's specific component-aware + SDS + SR pipeline.

## Re-check trigger

Re-run this Phase 2 step if any of the following happens:

- `github.com/grovessss/HFH-Font` gets a non-README commit (poll quarterly).
- Author releases code via a different org (search `Zhouhui Lian font diffusion code 2026` periodically).
- A reproduction repo from a third party appears via paperswithcode / arxiv comments.
