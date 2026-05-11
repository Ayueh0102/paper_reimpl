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
| GPU 數量 | **2 張**（user 原本估「單張」、實際是雙張）|
| GPU 型號 | NVIDIA RTX 6000 Ada Generation |
| VRAM | 48 GB / 卡（49140 MiB）|
| CUDA Driver | 573.42 |
| CUDA 版本 | 12.8 |
| GPU Mode | WDDM（Windows 一般用，非 TCC）|

**注意**：GPU 1 audit 當時被別人佔用（100% util，39.5 GB 已用，PID 26156 anaconda3 python.exe + PID 16524 anaconda3/envs/tools python.exe）。**我們訓練先綁 `cuda:0`**，並在訓練前 `nvidia-smi` 確認。

LM Studio 兩張卡都有 inference process，但只用 reserved memory，不影響我們訓練（pinning 後 LM Studio 會自動避讓）。

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
