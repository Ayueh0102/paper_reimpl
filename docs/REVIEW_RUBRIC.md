# Review Rubric

96 道 review gate 決策的判斷依據。Reviewer 一律輸出三種 verdict：

- **PASS** — 沒問題或非常小的 nit 已 inline 改
- **PASS-WITH-NITS** — 有發現問題但不阻塞下階段；列在報告底部 backlog
- **FAIL** — 必須先修才能進下階段；列出具體 fix（檔案、行號、預期改動）

---

## DL Reviewer Checklist

Reviewer 必須讀 `paper_notes/<NN>.md`（理解論文意圖）+ 對應 phase 的 src code，然後逐項勾選。

### Loss 正確性

- [ ] **loss 公式對應 paper**：寫進報告，把論文 eq.N 跟 code 對應行貼出來
- [ ] **reduction 模式**：`reduction='mean'` vs `'sum'` 跟論文一致（差一個 1/B 因子會讓 lr 表現完全錯）
- [ ] **權重組合**：multi-term loss 的權重值跟 paper Table N 一致
- [ ] **diffusion target**：`x0 / epsilon / v` 預測目標跟論文一致；schedule 對齊
- [ ] **CFG dropout**：若 paper 用 classifier-free guidance，訓練時 condition dropout p 對齊

### Gradient 流

- [ ] **無 detach 漏掉**：critical path 上沒有意外 `.detach()` 或 `torch.no_grad()`
- [ ] **conditioning 路徑連通**：char_id / writer_id / IDS 條件 embedding 真的影響到 loss
- [ ] **EMA / weight init**：論文要求的 EMA、init scheme（如 zero-init for AdaLN）有實作
- [ ] **grad clip**：若論文用 `clip_grad_norm_`，數值對齊

### Schedule & Sampler

- [ ] **β schedule**：linear / cosine / 其他 — 與論文一致；β_t 單調、α_T ≈ 0
- [ ] **sampler 對應 noise convention**：訓練是 ε-prediction，sampler 也用 ε-prediction（不可混 x0/ε）
- [ ] **time embedding**：sin/cos positional + MLP；維度對齊
- [ ] **訓練/推論 step 數一致性**：若論文訓練 1000 step、推論 50 step DDIM，文件有交代

### Data normalization

- [ ] **image range**：論文用 [-1, 1] 我們也是；不要混 [0, 1]
- [ ] **augmentation**：與 paper 一致；尤其書法不能水平翻轉、不能大幅 rotate
- [ ] **content cache 對齊**：bitmap / SDF / skeleton 軸與論文符號學一致

### Conditioning paths（每篇 paper 特化）

| Paper | 條件路徑必檢查 |
|---|---|
| FontDiffuser | style encoder 出來的特徵真有進 cross-attention；MCA 各 scale 都有用 |
| HFH-Font | component attention 的 keys/values 來源是 reference glyph，不是 target |
| IF-Font | IDS sequence tokenize 後真的進 text encoder；VQ codebook 索引在合理範圍 |
| VQ-Font | VQGAN codebook 凍結；transformer 預測的是 codebook index 不是 raw pixel |
| QT-Font | quadtree 的階層結構真有對應到 graph node attention |
| Calliffusion | BERT encoder freeze（前期）vs trainable（finetune）切換對；LoRA rank 對 |
| Moyun | TripleLabel 三個 embedding 真的各自獨立（不是共用 weight）；Mamba state 連續 |
| DP-Font | PINN 物理 loss 真有反向傳回 generator；殘差項是論文公式 |

### Training dynamics

- [ ] **loss 沒有 NaN**：smoke test 跑 10 step 不出 NaN
- [ ] **EMA 不會 decay 過快**（典型 0.999 / 0.9999）
- [ ] **batch size 與 lr scaling**：原 paper batch 256，我們 batch 16 + grad_accum 16 等於 256；lr 用 linear rule
- [ ] **建議 ablation**：列 1-2 個應該跑的消融（不一定要做，標 nice-to-have）

### Verdict 寫法（DL）

```markdown
# DL Review — <NN>_<paper> — <phase>

## Verdict: PASS / PASS-WITH-NITS / FAIL

## Checked
- [✓] loss formula matches paper eq.5 (src/model.py:142)
- [✓] gradient flow intact (no detach on critical path)
- [✗] **FAIL**: conditioning path broken — writer_id embedding not added to time-mixed feature, fix in src/model.py:88

## Suggested fixes (FAIL items)
1. src/model.py:88 — insert `h = h + writer_emb` before AdaLN block
2. configs/train_stage_b.yaml — `condition_dropout: 0.1` (currently 0.0, paper §3.2 requires 0.1 for CFG)

## Nice-to-have (PASS-WITH-NITS)
- 補一個 sanity test 驗證 EMA 數值

## Suggested ablations (optional)
- writer_dropout=0 vs 0.1 vs 0.3 on Stage B
```

---

## Code Reviewer Checklist

主要 invoke `everything-claude-code:python-review`，但補一份本 repo 特化的檢查：

### Public-repo 安全

- [ ] **無 SSH 帳密 hardcode**：所有 SSH host / user / password 從 env vars (`LAB_SSH_*`) 讀；不要寫死任何 IP、用戶名、密碼字串
- [ ] **無 secret token**：`.env` 在 gitignore；無 API key 字串
- [ ] **無內部資料路徑**：grep `D:\Char\ayueh` 僅出現於 docs 或 bat（不在 .py）

### 可重現性

- [ ] **seed 設定**：`train.py` 開頭 set `torch.manual_seed`, `np.random.seed`, `random.seed`
- [ ] **determinism flag**：若論文要求 `torch.use_deterministic_algorithms`，有設
- [ ] **no silent except**：`except:` / `except Exception: pass` 一律 FAIL

### Backend portability

- [ ] **無 hardcoded path**：所有資料路徑透過 `paper_reimpl_shared.data.manifest` 解析
- [ ] **無 `mother_repo_link/` 直接出現於 .py**：只能透過 manifest backend
- [ ] **device flag 從 CLI 來**：不寫死 `cuda:0`
- [ ] **Stage launcher 無 hardcoded SSH target/密碼**：shell scripts 也必須從
  `LAB_SSH_HOST / LAB_SSH_USER / LAB_SSH_PASS` 或呼叫端參數讀，不可寫死 IP、
  username、password。
- [ ] **Windows supervisor regex 可匹配真實 log**：PowerShell `-match` pattern
  中的 Windows path 只能用 regex 需要的單層 escaping；不要把
  `D:\Char\...` 寫成會匹配雙反斜線的 pattern。每個 supervisor 要有
  `Traceback` fail-fast branch。

### Coding hygiene

- [ ] PEP 8 (ruff check passes)
- [ ] type hint at all public APIs（dataset, model, train fn）
- [ ] docstring 至少有一行說明 module 在做什麼
- [ ] `__all__` 設定（若該模組有 export）
- [ ] 沒有 commented-out 大塊 code

### Testability

- [ ] `tests/test_smoke.py` 真的測 model forward + backward + 1 optimizer step
- [ ] smoke test 不依賴真實資料（用 `--synthetic` 或 tiny random tensor）
- [ ] config schema 有 dry-run validate；repo contract 的
  `paper_reimpl_shared.runner.entrypoint --dry-run --synthetic` 必須真的完成
  1 個 optimizer step，不能只建模後 exit 0。

### Data / training readiness

- [ ] **VQ codebook movement**：03/04 這類 VQ paper 必須有 CPU test 驗證
  codebook/embedding 在 Stage 0 one-step optimizer update 後有變動，避免
  `token_ce` 下降但 decoder 仍使用初始化 codebook。
- [ ] **ref-image conditioning data 非空**：01/02/04/05 等 ref-conditioned paper
  的 real manifest 必須含 `ref_image_paths`，且 reference 應同 writer 或同
  unit、排除 query char；空 list 只能在 synthetic/dry-run 使用。
- [ ] **writer imbalance 有處理**：Stage B/C 若 writer row count 差距很大，
  DataLoader 必須使用 writer-balanced sampler、per-writer cap，或在 report
  中明確標註稀有 writer embedding 會欠訓練。
- [ ] **paper-inspired vs faithful 標籤**：若 loss/conditioning 是 surrogate
  （例如 DP-Font PINN PDE 未知時的 Laplacian/TV/speckle surrogate），report
  必須寫明不是 faithful reproduction，且訓練表格不能拿來直接對 paper headline
  number 做同義比較。

### Per-paper assignment（2026-05-19 sign-off）

| Paper | Mode | Reason |
|---|---|---|
| 01 FontDiffuser | **faithful 嘗試** | 最常被引用 baseline；ref-image cross-attn 結構單純可逼近 |
| 02 HFH-Font | paper-inspired | latent + VAE decoder fine-tune 細節未公開 |
| 03 IF-Font | paper-inspired | IDS tokenizer + AR transformer 跟官方差距大 |
| 04 VQ-Font | paper-inspired | codebook + SSEM 細節差，但要等 Stage 0 重訓 |
| 05 QT-Font | paper-inspired | octree topology / D3PM 跟官方差太多 |
| 06 Calliffusion | **faithful 嘗試** | BERT + LoRA 結構單純，code 已乾淨 |
| 07 Moyun | paper-inspired | latent path + VAE 同 02；MAMBA backbone 取 fallback |
| 08 DP-Font | paper-inspired | **PINN 三項 loss 全是 surrogate**，PDE 未公開；明示不對應 paper headline number |

### Verdict 寫法（Code）

```markdown
# Code Review — <NN>_<paper> — <phase>

## Verdict: PASS / PASS-WITH-NITS / FAIL

## Auto check
- ruff check: PASS
- mypy: PASS-WITH-NITS (3 missing return type hints)
- pytest: PASS

## Manual check
- [✓] no hardcoded SSH creds (grep clean)
- [✗] **FAIL**: src/dataset.py:33 hardcodes `mother_repo_link/data/...`, should go through manifest backend
- [✓] type hints at public APIs

## FAIL fixes
1. src/dataset.py:33 — use `manifest.resolve_path(rel)` instead

## Nits
- 3 missing return type hints (mypy report attached)
```

---

## 簡化版（Stage A/B/C DL review）

Phase 3 訓練中的 stage DL review 不重複 deep dive，只看：

- [ ] 訓練曲線 loss 單調下降（或合理震盪）；最後 1k step 不發散
- [ ] 無 NaN / Inf（grep log）
- [ ] sample grid 至少能看出字（不是 noise）；風格 vs 內容大致對
- [ ] ckpt hash 記錄、檔案大小合理
- [ ] 跟 paper headline number 在合理 ballpark（±50% 接受 stage A，stage C 收斂後 ±20%）

verdict 同上。
