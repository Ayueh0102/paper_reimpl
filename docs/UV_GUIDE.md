# uv 使用指南

每篇 paper 是獨立 uv project，避免 8 篇依賴衝突（Mamba/diffusers/transformers/torch 版本差異大）。

## 安裝 uv（一次性，每台機器）

```bash
# Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Lab server (Windows)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 第一次建立 paper 的 venv

```bash
cd papers/01_fontdiffuser
uv venv --python 3.11                    # 建 .venv/
uv pip install -e ../../shared           # 引入共用 mini-package paper_reimpl_shared
uv pip install -e .                      # 引入該 paper 自己的套件
uv lock                                  # 鎖定版本，產生 uv.lock（commit 進 git）
```

## 從 git clone 後重建 venv

`uv.lock` 已 commit，其他人 / lab server clone 後一行：

```bash
cd papers/01_fontdiffuser
uv sync                                   # 完整重建 .venv，從 uv.lock 恢復
```

## 跑東西

一律用 `uv run`，會自動 activate 該 paper 的 venv：

```bash
uv run python -m paper_reimpl_shared.runner.entrypoint --paper fontdiffuser ...
uv run pytest tests/
uv run python src/fontdiffuser/sample.py --ckpt outputs/stage_a/...
```

不要 `source .venv/bin/activate`（雖然也能用，但 `uv run` 比較不會混到其他 venv）。

## 加新依賴

```bash
cd papers/01_fontdiffuser
uv add diffusers==0.27.0                  # 自動加入 pyproject.toml + 更新 uv.lock
uv add --dev pytest-mock                  # dev only
```

## 兩個 paper 都用到同一個 lib

每篇各自加，不要試圖跨 paper 共用 venv。`uv.lock` 各自獨立才能跨機器復現。

`shared/` 是個 mini-package 內含 `paper_reimpl_shared`，每篇都 `pip install -e ../../shared` 引入，所以 `shared/` 改了之後每個 paper 都自動拿到（editable install）。

## 跨機器 sync（git → server）

Lab server 上：

```bash
cd D:\Char\ayueh\paper_reimpl
git pull
cd papers\01_fontdiffuser
uv sync                                   # 重建 venv 對齊 uv.lock
```

## 故障排除

### `uv venv --python 3.11` 找不到 3.11
uv 會自動下載 Python 3.11，不用本機已裝。

### `mamba-ssm` Windows 安裝失敗（Moyun 會遇到）
fallback 純 PyTorch mamba：
```bash
uv remove mamba-ssm
uv add mamba-ssm-pytorch  # 或 cherry-pick from mamba.py
```

### shared 改了但 paper 沒看到變化
`uv pip install -e ../../shared` 是 editable，理論上 import 後 reload 就有；若用 IPython/Jupyter 要 `%load_ext autoreload`。

### `uv.lock` 在 PR 衝突
重新 `uv lock`，commit。lock 衝突是預期的、可重做的。

## .gitignore 約定

```
papers/*/.venv/          # uv 自動建，不 commit
papers/*/uv.lock         # commit！跨機器復現必要
shared/.venv/            # 如果單獨 dev shared，gitignored
```
