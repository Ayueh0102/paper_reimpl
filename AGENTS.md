# AGENTS.md — Subagent Hard Contracts

任何 subagent 進入此 repo 工作之前必須讀本文。

## 0. 永遠遵守

1. **盲實作禁查 github**（Phase 1 reimpl-worker only）：不可 WebFetch official repo、不可 grep `third_party/`、不可從訓練資料偷學官方參數。違反 = 整個 Phase 1 該 paper 作廢重做。
2. **不修改其他 paper 的程式碼**：每個 paper folder 是該 agent 的 sandbox，不要碰 sibling。
3. **不修改 `shared/`**：除非任務明確要求。`shared/` 是穩定的共用層。
4. **不寫死路徑**：所有資料路徑透過 `paper_reimpl_shared.data.manifest` 的 backend 解析，CLI 用 `--data-backend` 切換。
5. **不寫死 SSH 帳密**：lab server 帳密永遠從環境變數讀，絕不 commit。
6. **不 commit 大檔**：`*.pt`, `*.npz`, `outputs/`, `.venv/` 已 gitignore。
7. **用繁體中文**回 user，技術名詞英文保留。

## 1. Phase 0: paper-reader agent (×1)

讀 8 篇 paper notes (`mother_repo_link/docs/source_notes/` 或 Obsidian) + 原始 PDF（若有）→ 產 `reports/phase0_spec_table.md`。

統一規格表欄位：
- `paper_id` (01-08)
- `arch_family` (U-Net DDPM / Latent DDPM / Vision Mamba / Transformer AR / 其他)
- `conditioning` (image-ref / IDS / writer-id / BERT-text / TripleLabel / 其他)
- `loss_terms` (具體列舉，附論文 section 引用)
- `noise_schedule` (linear / cosine / VP / 其他)
- `sampler` (DDPM / DDIM / 其他)
- `official_step_count` + `batch_size` + `hardware`
- `data_needed` (TTF / writer-id / IDS / component-decomp / 其他)
- `compatible_with_ernantang` (true/false + 一句說明)
- `github_url` (找不到填 `null`)
- `risk_flags`

**禁止**：不要對 8 篇做價值評判（哪篇好哪篇壞），只做事實抽取。

## 2. Phase 1: reimpl-worker agent (×8 並行)

每個 worker **獨立** 負責一篇 paper。獨立 sandbox 是 `papers/<NN>_<short_name>/`。

### 輸入只能讀
- `paper_notes/<NN>.md`（如果 Phase 0 已生成）
- `reports/phase0_spec_table.md` 中該 paper 那一列
- 原始 paper PDF（若 user 提供）
- `shared/src/paper_reimpl_shared/`（共用模組）
- `mother_repo_link/data/ttf_renders/` 的 sample 結構（不看實際 image）

### 輸入禁止讀
- `third_party/`（Phase 2 才會 clone official）
- WebFetch / WebSearch 對 github / huggingface 的 official 實作
- sibling paper folder

### 輸出
1. `paper_notes/<NN>.md` — 自己理解後的架構/loss/data flow 抽象筆記
2. `src/<paper>/{model.py, train.py, sample.py, dataset.py, configs/{model.yaml, train_stage_a_ttf.yaml, ...}}` — 能 run
3. `tests/test_smoke.py` — `uv run pytest -x` 通過
4. `reports/blind_impl.md` — 決策日誌：每個非平凡決策標 `[paper-cited section X.Y]` 或 `[guessed-because-paper-vague]`
5. `pyproject.toml` + `uv.lock` — 該 paper 專屬依賴

### 完成定義（gate 1）
- `uv sync` 通過
- `uv run pytest tests/test_smoke.py -x` 綠燈
- `uv run python -m paper_reimpl_shared.runner.entrypoint --paper <paper> --dry-run --synthetic --device cpu --train ... --model ... --data ... --data-backend mac_symlink` 跡 1 step 不崩
- `reports/blind_impl.md` 包含至少 5 條 `[guessed-...]` 條目（驗證有思考、不是抄）

## 3. Phase 1→2 Gate: dl-reviewer + code-reviewer (×8 對)

詳細 checklist 見 `docs/REVIEW_RUBRIC.md`。

簡述：
- **dl-reviewer**：loss 對不對、gradient 有沒有斷、條件路徑通不通、schedule 合不合理、sampler 跟訓練 noise 是否一致、data norm 對齊論文
- **code-reviewer**（用 `everything-claude-code:python-review`）：PEP 8、type hint、無 hardcoded path、seed 有設、無 silent except、公開 repo 安全

兩 reviewer 都回 `{PASS, PASS-WITH-NITS, FAIL}`。**FAIL 阻斷** 進 Phase 2；reimpl-worker 修完再簽。

## 4. Phase 2: github-diff agent (×8 並行)

職責：
1. 找 official repo（從 `reports/phase0_spec_table.md` 的 `github_url`；若 null 用 paper title arxiv → PapersWithCode）
2. `git clone` 到 `third_party/<NN>_<paper>/`
3. 對 blind impl 做 diff，產 `reports/github_diff.md`，分類：
   - `arch_deltas` — 架構差異（U-Net depth、attention head、embed dim）
   - `loss_deltas` — loss 公式 / reduction / weighting 差異
   - `schedule_deltas` — β_t、time embedding、sampler 步驟
   - `conditioning_deltas` — 條件注入點、embedding 維度、CFG hooks
   - `hparam_deltas` — lr、optimizer、batch、grad clip
   - `data_pp_deltas` — 圖像 norm、augmentation、字典建立
   - `risk_of_bug` — 我們 blind impl 看起來有 bug 的地方

**禁止**：不要直接改 reimpl 的 code。寫 diff 報告，交還 reimpl-worker 修正。

若 official 找不到：`reports/github_diff.md` 開頭寫 `STATUS: official_unavailable`，DL reviewer 接手做更嚴格 review。

## 5. Phase 3: train-launcher agent (×1, 序列)

從 Phase 2 通過的 paper 依序：
1. SSH `nvidia-smi` 確認 cuda:0 空閒
2. 啟動 Stage A（TTF）bat
3. `ScheduleWakeup` 30 分鐘 poll log；偵測 NaN/crash/完成
4. 拉回 `outputs/stage_a/summary.json` + `sample_grid.png` 到 Mac
5. 觸發 stage A DL review → 通過進 Stage B → ... → Stage C → final_results

## 6. 工具使用約定

- 任何 file edit 用 `Edit`，新檔用 `Write`
- 跑 shell 用 `Bash`；長時間訓練用 `Bash run_in_background=true`，搭 `Monitor` tail log
- 平行 agent 用 `superpowers:dispatching-parallel-agents`
- 規劃用 `superpowers:writing-plans`
- TDD 用 `superpowers:test-driven-development`
- 提交前用 `superpowers:verification-before-completion`
- 任何修改用 `TaskCreate`/`TaskUpdate` 追蹤

## 7. 違反 contract 的處理

主 agent（user 對話的那個）若發現 subagent：
- 偷查 github（Phase 1）→ 整個該 paper Phase 1 重做
- 修了 `shared/` → revert，叫 subagent 改提 PR
- 寫死路徑 → 退回修正
- commit 大檔 → `git rm --cached` + 修 `.gitignore`
