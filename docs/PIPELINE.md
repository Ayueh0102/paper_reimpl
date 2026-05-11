# Pipeline 詳細流程

完整 4-phase + 雙 review gate workflow。每個 subagent 進來先讀 `AGENTS.md`，再讀本文。

## 流程圖

```
              ┌─────────────────────────────────────────────────────────┐
              │ Phase 0  paper-reader (×1)                              │
              │   讀 8 篇 → reports/phase0_spec_table.md                │
              └─────────────────────────────────────────────────────────┘
                                       ↓
              ┌─────────────────────────────────────────────────────────┐
              │ Phase 1  reimpl-worker × 8 (並行) - 盲實作              │
              │   每 worker: paper_notes / src / configs / tests        │
              │   交付 reports/blind_impl.md                            │
              └─────────────────────────────────────────────────────────┘
                                       ↓
                          ┌────────────────────────────┐
                          │ Gate 1: blind review (×8)  │
                          │   dl_review_blind.md       │
                          │   code_review_blind.md     │
                          └────────────────────────────┘
                                       ↓
              ┌─────────────────────────────────────────────────────────┐
              │ Phase 2  github-diff × 8 (並行)                          │
              │   clone official → reports/github_diff.md               │
              │   reimpl-worker × 8 修正 code                            │
              └─────────────────────────────────────────────────────────┘
                                       ↓
                          ┌─────────────────────────────────┐
                          │ Gate 2: post-diff review (×8)   │
                          │   dl_review_post_diff.md        │
                          │   code_review_post_diff.md      │
                          └─────────────────────────────────┘
                                       ↓
              ┌─────────────────────────────────────────────────────────┐
              │ Phase 3  train-launcher (序列, 單 GPU)                  │
              │                                                         │
              │   Stage A: TTF pretrain (paper original step × 0.5-1.0) │
              │     ↓ Gate 3: dl_review_stage_a.md                      │
              │   Stage B: 多家書法 mid-train (5-20k step)              │
              │     ↓ Gate 4: dl_review_stage_b.md                      │
              │   Stage C: 二南堂 finetune (3-10k step)                  │
              │     ↓ Gate 5: dl_review_stage_c.md                      │
              │                                                         │
              │   reports/final_results.md (含與 A3.15 對比 grid)       │
              │     ↓ Gate 6: final dl_review + code_review             │
              └─────────────────────────────────────────────────────────┘
```

8 papers × 6 gates × 2 reviewer = **96 reports** 預估上限。但 stage A/B/C 的 DL review 可用 lightweight template（看 loss 曲線 + 1-2 個 sample），不必每篇都做 deep dive。

## Gate 通過標準

| Gate | DL reviewer 必過 | Code reviewer 必過 | 其他 |
|---|---|---|---|
| 1. blind | arch 對 paper、loss 公式對、gradient 連通、no NaN at smoke | PEP 8、type hint、無 hardcoded path、無 silent except、seed 設定 | `pytest -x` 綠 |
| 2. post-diff | 已吸收 github_diff 高優先項、無回歸 | 同上 | `pytest -x` 仍綠 |
| 3. stage A | 訓練曲線下降、最後 1k step 不發散、TTF sample 像字 | （不重複 review code，除非有改）| `nvidia-smi` 無 OOM record |
| 4. stage B | 多家書法 sample 看得到 writer style 差異 | 同 3 | ckpt hash 記錄 |
| 5. stage C | 二南堂 sample 接近 A3.15 v4_full 品質 | 同 3 | 同 4 |
| 6. final | 與 paper headline number 對比合理（±20%）| 全 repo lint 通過 | final_results.md 完整 |

FAIL → reimpl-worker 修；PASS-WITH-NITS → 記錄 nits 進 backlog 繼續；PASS → 進下一階段。

## 時程與平行度

| 階段 | 預估 | 平行度 |
|---|---|---|
| Phase 0 | 0.5 day | 1 agent |
| Phase 1 blind impl | 2 days | 8 agent 並行 |
| Gate 1 (×8) | <1 day | 8 對 reviewer 並行 |
| Phase 2 diff + fix | 2 days | 8 agent 並行 |
| Gate 2 (×8) | <1 day | 8 對 reviewer 並行 |
| **Phase 0-2 小計** | **~5 days** | 主要在 reviewer 周轉 |
| Phase 3 (8 paper × 3 stage) | 16-32 GPU-days | 序列 |

## Pipeline Shakedown 順序

第一篇做 **01_fontdiffuser**（U-Net DDPM 基線、github 已知、無特殊依賴），全程跑完 Phase 0-3 Stage A 後檢視：
- shared/ 模組有沒有缺
- template 有沒有要改
- bat 腳本對不對

再啟動 02-08 並行。

## 各 phase agent 模板位置

- Phase 0: `skills/paper-reimpl-pipeline/templates/paper_reader_prompt.md`
- Phase 1: `skills/paper-reimpl-pipeline/templates/reimpl_worker_prompt.md`
- Phase 1/2 gate DL: `skills/paper-reimpl-pipeline/templates/dl_review_prompt.md`
- Phase 1/2 gate code: 直接 invoke `everything-claude-code:python-review`
- Phase 2: `skills/paper-reimpl-pipeline/templates/github_diff_prompt.md`
- Phase 3: `skills/paper-reimpl-pipeline/templates/train_launcher_prompt.md`

## Report 統一輸出位置

每篇 paper 的 reports：
```
papers/<NN>_<short>/reports/
├── blind_impl.md             # Phase 1 worker
├── dl_review_blind.md        # Gate 1
├── code_review_blind.md      # Gate 1
├── github_diff.md            # Phase 2
├── dl_review_post_diff.md    # Gate 2
├── code_review_post_diff.md  # Gate 2
├── stage_a_results.md        # Phase 3 stage A
├── dl_review_stage_a.md      # Gate 3
├── stage_b_results.md
├── dl_review_stage_b.md      # Gate 4
├── stage_c_results.md
├── dl_review_stage_c.md      # Gate 5
└── final_results.md          # Gate 6 + 跟 A3.15 對比
```

頂層 `reports/phase0_spec_table.md` 是跨 8 篇統一表（Phase 0 唯一輸出）。
