# Stage A TTF Pretrain — 實驗總結（2026-05-14）

8 篇 paper 在 13 OFL TTF 字體 cross-font pretrain 任務上的 reimpl 現況。所有
sample 用 12×3 grid 視覺判斷（左：source / 中：GT target font / 右：generated）。

## 一張表看完

| # | Paper | Stage A 最佳設定 | Sample 品質 | 結論 | 訓練手段 |
|---|---|---|---|---|---|
| 01 | **FontDiffuser** | v3 80k step | **8-9/12 可讀** | ✅ 可用 baseline | ref image style transfer |
| 02 | **HFH-Font** | v3 50k step（frozen pretrained VAE）| 2-3/12 emerging structure | ⚠️ 部分 | latent diffusion |
| 03 | **IF-Font** | v2 200k step | 純灰塊（sq=0 collapse） | ❌ 失敗 | VQGAN + IDS + AR Transformer |
| 04 | **VQ-Font** | v3 chain 訓中（VQGAN 50k + TFM 200k）| TBD（v2 missing=222 失敗）| 🔄 重訓中 | VQGAN + Transformer prior |
| 05 | **QT-Font** | v2 30k @ depth=6 + sparse + bs=4 | 64px pixel-art 可辨 | ✅ 可用 | quadtree D3PM categorical |
| 06 | **Calliffusion** | v2 200k | 出 Chinese 字但 **char identity 錯** | ⚠️ prompt 問題 | DDPM + BERT prompt |
| 07 | **Moyun** | 未實作 | — | 📝 待補 | Vision Mamba + TripleLabel |
| 08 | **DP-Font** | v2 200k → v3 訓中 | **11/12 可讀，最強** | ✅✅ 最強 | DDPM + writer_id + multi-attr |

## 每篇優勢與限制

### 01 FontDiffuser ✅
- **架構**：pixel-space DDPM U-Net，128×128，content encoder + style encoder
- **Style 條件**：傳入 ref 字體**整張圖**（真實 spatial features）
- **優勢**：style transfer 機制清楚、訓練穩、結果可讀
- **限制**：需要 ref 圖，無法純 prompt 控制
- **訓練要訣**：v1 5k → v3 80k 質的躍升；單純加 step 數可救

### 02 HFH-Font ⚠️
- **架構**：latent diffusion（128×128 → 16×16 latent via VAE down_factor=8）
- **獨特處**：**唯一用 VAE compression 的 paper**
- **關鍵教訓**：
  - v1/v2 隨機 VAE → latent collapse → 灰糊／黑白雜訊
  - **v3 必須先 pretrain VAE（30k step recon）→ 凍結 → 訓 U-Net 50k step**
  - v4 warm-start 用 lr=1e-4 + OneCycleLR ramp → 破壞 v3 收斂的權重（regression）
  - v5 改 lr=1e-5 → 不破壞但也沒進步
- **結論**：50k step 已近上限，下一步應該換 Stage B 真資料

### 03 IF-Font ❌
- **架構**：VQGAN (frozen) → IDS tokens → AR Transformer (sup_cl + sq CE loss)
- **失敗模式**：
  - paper 預設用 CompVis vq-f8-n256 預訓，我們沒有 → 自己 pretrain VQGAN 30k step（active_codes=253/256 看似 OK）
  - 但 Phase-2 AR 訓 200k 後 **sq loss 收斂到 0.0000** → token collapse
  - AR 學會「永遠輸出背景 token 序列」（128px 字面 80%+ 是背景，CE 最小化 = 背景單調預測）
  - cl 持平 5.0 但跟生成解耦
- **sample 結果**：純灰色方塊
- **可能的修法**：sq CE 加 background mask、或下載 CompVis 真貨
- **目前狀態**：擱置

### 04 VQ-Font 🔄
- **架構**：custom VQGAN（K=1024, 16×16 latent）+ Transformer prior + structure_id 條件
- **v2 失敗**：訓練只跑 transformer stage，沒先 pretrain VQGAN → 載入時 missing=222 → 推論 garbage（純灰塊）
- **v3 修法**：先 Stage 0 VQGAN 50k（simple_loss recon only，跳過 LPIPS+GAN 提速）→ 再 Stage Transformer 200k 用 vqgan_ckpt warmstart
- **預期收穫**：v3 有真 codebook，sample 才能出真實 glyph

### 05 QT-Font ✅
- **架構**：quadtree D3PM uniform diffusion
- **獨特處**：**8 paper 中唯一用 categorical state（K=3: bg/contour/skeleton）**，最接近離散值生成
- **OOM 救援**（重大工程修）：
  - paper 原 depth=7 (128px) bs=4 OOM at 48 GB → 11 GB/sample
  - 根因：`EdgeTypedGraphConv` forward 把 `W[edge_type]` gather 成 `(E, in, out)` 巨大張量；E=64k 時光這一個張量 32 GB
  - **修法 1**：降 depth 7→6（128→64 px），node 數 21k→5k
  - **修法 2**：**改 sparse EdgeTypedGraphConv**——4 個 edge_type 各自 `(E_t, in) @ (in, out)`，**peak mem 降 ~512×**
  - 兩個一起 → bs=4 跑得起來，速度比 v1 快 5×
- **結果**：30k step at depth=6 → 64×64 pixel-art 字形可辨
- **限制**：64×64 太小，字形粗糙；若要 128×128 需要更大記憶體或更多 sparse 優化

### 06 Calliffusion ⚠️
- **架構**：DDPM U-Net + BERT text encoder + LoRA（per-writer）
- **目前問題**：
  - **使用 StubTextEncoder（use_bert=false）** → BERT 語意完全沒用上
  - prompt 格式 `"量 kai noto_serif_sc"` 後段英文字體名 stub encoder 學不起來
  - v2 sample：模型畫出 Chinese 字，**但跟 prompt 的目標字不對齊**（char identity 錯）
- **v3 計畫**：
  - `use_bert=true` 啟用真 BERT
  - prompt 改中文：`"量 楷書 顏體"`（用 BERT 認識的中文 token）

### 07 Moyun 📝
- **架構**：Vision Mamba + TripleLabel（字／書家／字體類別三標籤）+ LoRA
- **狀態**：尚未實作
- **風險**：`mamba-ssm` Windows 安裝可能困難

### 08 DP-Font ✅✅
- **架構**：pixel-space DDPM at 80×80 + multi-attribute guidance
- **多屬性 conditioning**：
  - writer_id（整數 ID，**非 reference image**）
  - script_id, char_id（categorical）
  - stroke_order（hash 合成，paper 原意是真筆順）
  - ink_intensity, font_size（hash 合成）
  - PINN loss（ink-diffusion / nib-motion / continuity）**權重 0 關掉**
- **fixed-source 驗證**：固定 source=sans 但 writer_id 變化，sample 仍**忠實切換風格**（草書 writer_id → 草書 output）→ style disentanglement 真的學起來了
- **為什麼結果最好**：
  1. bs=16 + 200k step（v3 加到 400k 中）
  2. content encoder 提供 layout 監督（dense per-stage fusion）
  3. 80×80 小尺寸 → 模型容量易塞滿
  4. multi-attribute embedding 之中 writer_id 真的學會 style mapping
- **限制**：
  - stroke_order 是假的（hash 合成）—— 未來補真筆順可能再進步
  - 沒用 ref image → 不能輕易擴充到沒見過的 writer

## 工程基礎建設教訓（適用全部 paper）

### 1. DataLoader 性能（2026-05-14 修）
所有 paper 原本只設 `num_workers`，沒設：
- `persistent_workers=True`（避免 epoch 邊界 worker 殺重啟）
- `pin_memory=True`（host→device 加速）
- `prefetch_factor=4`（每 worker 預先 fetch 4 batch）

**修後 GPU 利用率從 30-55% 預期上看 70-90%**。

### 2. bs/nw 預設規格
- 02/04/06/08: **bs≥32, nw=4-8**
- 05 quadtree: **bs=1-4 例外**（記憶體上限）

### 3. Warm-start LR
- 從已收斂 ckpt continuation 必須**降 LR 10× 以上**
- 02 v4 失敗教訓：v3→v4 用 lr=1e-4 + OneCycleLR 把收斂權重打散
- 02 v5 修：lr=1e-5（成功保留 v3）

### 4. schtasks Windows pitfall
- `/SC ONCE /ST <time>` 創建後即使手動 /Run，**到指定時間還會再自動 fire 一次**
- 03 v2 因此跑了兩次（同樣的 sq=0 結果）→ 浪費 ~10 GPU hour
- 教訓：trigger 後立刻 /End 或設極遠未來時間

### 5. Logger 配置一致性
- 03/04/05/06/08 有 5 種不同 logger setup
- 04 的 logger 從未掛 handler → 200k step 訓練全程 log = 0 bytes（驗證只能靠 ckpt 存在）
- 教訓：所有 paper 統一在 entrypoint 設 root logger basicConfig

## 排名（對二南堂 finetune 任務的可用度）

| Rank | Paper | 進 Stage B 信心 | 額外條件 |
|---|---|---|---|
| 1 | **08 DP-Font** | 高 | 風格 disentanglement 已驗證，可立刻 Stage B |
| 2 | **01 FontDiffuser** | 高 | v3 已 8/12 readable，style transfer 機制完整 |
| 3 | **05 QT-Font** | 中 | 可進 Stage B，但 64px 限制畫質 |
| 4 | **02 HFH-Font** | 中 | v3 emerging，Stage B 真資料應該能突破 |
| 5 | **06 Calliffusion** | 低 | 需先修 prompt schema（v3 計畫）|
| 6 | **04 VQ-Font** | 待定 | v3 訓中，需先看 sample |
| 7 | **07 Moyun** | 未知 | 尚未實作 |
| 8 | **03 IF-Font** | 低 | sq=0 collapse 擱置，需要 mask 或 CompVis VQGAN |

## 目前訓練狀態（2026-05-14 10:30）

- **cuda:0**: 08 v3 200k continuation（lr=1e-5 warm-start from v2）
- **cuda:1**: 04 v3 chain（VQGAN 50k → Transformer 200k）

兩者預計 6-8 hr 完成。
