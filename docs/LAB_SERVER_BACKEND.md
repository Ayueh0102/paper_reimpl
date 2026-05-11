# Lab Server Backend

實驗室 Windows server SSH audit 結果（2026-05-11 11:50）。**所有實際 IP / 帳密只存在本機環境變數，不進 repo。**

## 連線

```bash
# 設定本機環境變數（一次性）
export LAB_SSH_HOST="<set from local skill / .env>"
export LAB_SSH_USER="<lab user>"
export LAB_SSH_PASS="<lab pwd>"

# 之後就靠變數
sshpass -p "$LAB_SSH_PASS" ssh -o StrictHostKeyChecking=no "$LAB_SSH_USER@$LAB_SSH_HOST"
# Hostname: WIN-C20DRJGJ4S4 (Windows server, RFC1918 internal lab subnet)
```

帳密**不寫進 repo**。放在 `~/.claude/skills/lab-gpu-training/`（本機 skill）或 shell `.env`（gitignored）。

## 硬體

| 項目 | 值 |
|---|---|
| GPU 數量 | **2 張**（雙張可同時用）|
| GPU 型號 | NVIDIA RTX 6000 Ada Generation |
| VRAM | 48 GB / 卡（49140 MiB）|
| CUDA Driver | 573.42 |
| CUDA 版本 | 12.8 |
| GPU Mode | WDDM（Windows 一般用，非 TCC）|

**2026-05-11 狀態（user 確認）**：兩張 GPU 都已關閉佔用程序，皆閒置可用。
- GPU 0: 317 MiB used / 0% util ✓
- GPU 1: 5063 MiB used / 0% util ✓（LM Studio 常駐，不影響訓練）

## 平行訓練策略

48GB VRAM × 2 張意味著 Phase 3 可以**雙路平行**，不必純 sequential：

| 策略 | 同時跑 | 每模型 VRAM | wall-clock |
|---|---|---|---|
| Sequential (原 plan) | 1 paper | 整張卡 48GB | 16-32 days |
| **2-way parallel（推薦）** | 2 papers，一張卡一個 | 整張卡 48GB | **8-16 days** |
| 4-way parallel（激進）| 4 papers，每張卡塞 2 個 | 24GB/model | 4-8 days，但 CPU/IO 競爭風險 |

**預設**：2-way parallel（保守、易 debug）。FontDiffuser shakedown 跑完 Stage A 後，若 VRAM 還寬鬆且資料 I/O 不卡，再評估升級到 4-way。

## 軟體環境

| 項目 | 路徑 |
|---|---|
| uv | `C:\Users\Ptri\.local\bin\uv.exe` ✓ |
| git | `C:\Program Files\Git\cmd\git.exe` ✓ |
| Python | `C:\Users\Ptri\AppData\Local\Programs\Python\Python310\python.exe`（3.10）|
| Conda | `C:\Users\Ptri\anaconda3\` ✓ |
| 用戶名 | `Ptri` |

**uv 已預裝**，所以姊妹 repo 部署時直接 `cd papers/<NN> && uv sync` 即可。

## 磁碟

| Drive | 可用 |
|---|---|
| D: | **10.0 TB free**（總 10.9 TB） |

D: 大量可用空間，存資料 snapshot + checkpoints + logs 完全足夠。

## 工作目錄

```
D:\Char\ayueh\paper_reimpl\          ← git clone 到此
D:\Char\ayueh\paper_reimpl\data_snapshot\   ← scp 過去的 manifests/cache/ttf_renders
D:\Char\ayueh\paper_reimpl\logs\     ← 訓練 log
D:\Char\ayueh\paper_reimpl\papers\<NN>\outputs\  ← checkpoints, samples (gitignored)
```

`D:\Char\ayueh\` 已在 audit 時建立（先前不存在）。

## 編碼注意事項

Windows server 預設 CP950，cmd 輸出中文會亂碼。bat 檔頂部一律 `chcp 65001 >nul`，Python 程式設定 `PYTHONIOENCODING=utf-8`。SSH 拉取 log 時用 `powershell Get-Content -Encoding UTF8`。

## 已驗證 SSH 指令

```bash
SSH="sshpass -p \"$LAB_SSH_PASS\" ssh -o StrictHostKeyChecking=no $LAB_SSH_USER@$LAB_SSH_HOST"

# 連線測試
$SSH "echo OK && hostname"
# 預期：OK_FROM_LAB / WIN-C20DRJGJ4S4

# 磁碟空間
$SSH "fsutil volume diskfree D:"

# GPU 狀態
$SSH "nvidia-smi"
```

## 下一步（部署 sister repo）

1. SSH 進 server `mkdir D:\Char\ayueh\paper_reimpl` 並 `git clone <public-github-url> .`
2. Mac 端 `scp -r` data snapshot 過去 `data_snapshot/`
3. `cd papers\01_fontdiffuser && uv sync`
4. 第一次 stage A 跑 bat 確認 cuda:0 可用
