# Vast.ai Backend（stub）

未來把訓練搬到 vast.ai 的計畫。**目前不需要實作，留 stub 確保架構 portable。**

## 設計理念

`paper_reimpl_shared.data.manifest` 有三個 backend：

| Mode | 用途 | 路徑 base |
|---|---|---|
| `mac_symlink` | Mac 本地 | `mother_repo_link/...` |
| `lab_server` | 實驗室 | `D:\Char\ayueh\paper_reimpl\data_snapshot\` |
| `vast_snapshot` | vast.ai（未來） | `/workspace/data_snapshot/...` |

切換用 `--data-backend vast_snapshot`。**per-paper code 不變**，只改 backend。

## 預期 vast.ai 部署流程

1. 預建 Docker image（含 uv + Python 3.11 + CUDA 12.x base）推上 Backblaze B2 或 Docker Hub
2. vast.ai instance onstart script：
   ```bash
   apt update && apt install -y git
   git clone https://github.com/<user>/paper_reimpl /workspace/paper_reimpl
   cd /workspace/paper_reimpl/papers/01_fontdiffuser
   uv sync
   # 下載 data snapshot 從 B2
   curl -o /workspace/data_snapshot.tar.gz https://b2/.../snapshot.tar.gz
   tar xf /workspace/data_snapshot.tar.gz -C /workspace/
   ```
3. 跑訓練 `uv run python -m paper_reimpl_shared.runner.entrypoint --data-backend vast_snapshot ...`

## 需要做的（將來）

- [ ] `shared/src/paper_reimpl_shared/runner/launcher_vast.py` 實作
- [ ] Backblaze B2 snapshot 上傳腳本
- [ ] Docker image build + push
- [ ] vast.ai onstart 模板
- [ ] cost estimate 預算

## 觸發時機

當實驗室 server 不夠用、或需要更大 GPU（H100 / 8× A100）時。
