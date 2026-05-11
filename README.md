# Paper Reimpl Pipeline

盲實作 8 篇 diffusion-based 中文字／書法生成 paper，對齊 official github、跑三階段訓練（TTF → 多家書法 → 二南堂 finetune），把整套流程封裝成可重複的 skill。

姊妹於 [ernantang-jit-calligraphy-generation](https://github.com/Ayueh0102/ernantang-jit-calligraphy-generation)；此 repo 專做「文獻基線對照」與「方法學習」。

## 8 篇 Paper

| # | Paper | Venue | Design Pattern |
|---|---|---|---|
| 01 | FontDiffuser | AAAI 2024 | U-Net DDPM + style/content cross-attn |
| 02 | HFH-Font | SIGGRAPH-A 2024 | Latent DDPM + component attn + SR |
| 03 | IF-Font | NeurIPS 2024 | IDS + VQ tokens + AR Transformer |
| 04 | VQ-Font | AAAI 2023 | 預訓 VQGAN codebook + 12 結構類 |
| 05 | QT-Font | SIGGRAPH 2024 | Quadtree graph diffusion |
| 06 | Calliffusion | AAAI 2024 wksp | DDPM U-Net + BERT 文字 + LoRA |
| 07 | Moyun（墨韻）| ACM 2025 | Vision Mamba + TripleLabel |
| 08 | DP-Font | IJCAI 2024 | 書法 diffusion + PINN |

## 工作流（4 phase）

```
Phase 0  paper-reader  → 統一規格表
Phase 1  reimpl-worker × 8 (平行)  ── 盲實作，禁查 github
            ↓ DL + Code review 雙閘
Phase 2  github-diff × 8 (平行) → 對齊修正
            ↓ DL + Code review 雙閘
Phase 3  train-launcher (序列)  TTF → 多家書法 → 二南堂
            ↓ 每階段都過 DL review
         final_results.md (含與本 repo A3.15 對比)
```

詳見 `docs/PIPELINE.md`。

## 環境

每篇 paper 是**獨立 uv project**（避免 8 篇依賴衝突）：

```bash
# 安裝 uv（一次性）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 進入單篇 paper
cd papers/01_fontdiffuser
uv sync                                # 從 uv.lock 重建 .venv
uv run pytest tests/                   # 跑 smoke test
uv run python -m paper_reimpl_shared.runner.entrypoint --paper fontdiffuser ...
```

`shared/` 是公用 mini-package `paper_reimpl_shared`，被每篇 paper `uv pip install -e ../../shared` 引入。

## 訓練後端

| Backend | 用途 | 路徑 |
|---|---|---|
| `mac_symlink` | Mac 開發 | `mother_repo_link/...` |
| `lab_server` | 實驗室訓練 | `D:\Char\ayueh\paper_reimpl\data_snapshot\` |
| `vast_snapshot` | 未來雲端 | `/workspace/data_snapshot/...` |

實驗室規格見 `docs/LAB_SERVER_BACKEND.md`：2× RTX 6000 Ada 48GB、uv/git/conda 已預裝、D: 10TB 可用。

## 資料

每篇都跑三階段訓練：
- **Stage A** TTF 預訓練（13 fonts × 6500 字）— `shared/data/ttf_renders.py`
- **Stage B** 多家書法 mid-train（24 writer / 32 family / 84 unit）— 用 mother repo 的 character_disjoint split
- **Stage C** 二南堂 finetune — mother repo 的 a_main_clean_split_random 過濾二南堂 writer

## License

MIT
