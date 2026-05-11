"""Lab server bat-script emitter.

Generates `papers/<NN>/scripts/run_<NN>_stage_<x>_gpu<G>.bat` from a template.
SSH credentials are NEVER stored here — they come from environment variables:
  LAB_SSH_HOST, LAB_SSH_USER, LAB_SSH_PASS

Bat scripts themselves do not contain credentials (they run on the server).
The target server runs Windows, has uv pre-installed, and exposes
``D:\\Char\\ayueh\\paper_reimpl\\`` as the working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

BAT_TEMPLATE = r"""@echo off
chcp 65001 >nul
setlocal
set REPO=D:\Char\ayueh\paper_reimpl
set PR_DATA_ROOT=D:\Char\ayueh\paper_reimpl\data_snapshot
set PAPER_DIR=%REPO%\papers\{paper_dir}
set DATA=%PAPER_DIR%\src\{paper_pkg}\configs\{data_yaml}
set MODEL=%PAPER_DIR%\src\{paper_pkg}\configs\{model_yaml}
set TRAIN=%PAPER_DIR%\src\{paper_pkg}\configs\{train_yaml}
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=%REPO%\logs\{nn}_stage_{stage}_%DT%.log
cd /d %PAPER_DIR%
uv run python -m paper_reimpl_shared.runner.entrypoint ^
    --paper {paper_pkg} ^
    --train "%TRAIN%" --model "%MODEL%" --data "%DATA%" ^
    --data-backend lab_server --device cuda:{gpu} ^
    > "%LOG%" 2>&1
endlocal
"""


def emit_bat(
    *,
    paper_dir: str,        # "01_fontdiffuser"
    paper_pkg: str,        # "fontdiffuser"
    nn: str,               # "01"
    stage: str,            # "a" / "b" / "c"
    gpu: int = 0,
    data_yaml: str | None = None,
    model_yaml: str = "model.yaml",
    train_yaml: str | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Write a bat script under papers/<paper_dir>/scripts/. Returns the path."""
    data_yaml = data_yaml or f"data_stage_{stage}.yaml"
    train_yaml = train_yaml or f"train_stage_{stage}.yaml"
    content = BAT_TEMPLATE.format(
        paper_dir=paper_dir,
        paper_pkg=paper_pkg,
        nn=nn,
        stage=stage,
        gpu=gpu,
        data_yaml=data_yaml,
        model_yaml=model_yaml,
        train_yaml=train_yaml,
    )
    out_dir = out_dir or Path(__file__).resolve().parents[5] / "papers" / paper_dir / "scripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"run_{nn}_stage_{stage}_gpu{gpu}.bat"
    path.write_text(content, encoding="utf-8")
    return path


def ssh_command(remote_cmd: str) -> list[str]:
    """Build sshpass ssh command using env vars. Caller subprocess.run() this."""
    host = os.environ.get("LAB_SSH_HOST")
    user = os.environ.get("LAB_SSH_USER")
    pw = os.environ.get("LAB_SSH_PASS")
    if not (host and user and pw):
        raise EnvironmentError(
            "LAB_SSH_HOST / LAB_SSH_USER / LAB_SSH_PASS not all set; "
            "cannot SSH to lab server"
        )
    return [
        "sshpass",
        "-p",
        pw,
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        f"{user}@{host}",
        remote_cmd,
    ]
