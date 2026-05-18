@echo off
chcp 65001 >nul
setlocal
set REPO=D:\Char\ayueh\paper_reimpl\repo
set PAPER_DIR=%REPO%\papers\01_fontdiffuser
set DATA=%PAPER_DIR%\src\fontdiffuser\configs\data_stage_b.yaml
set MODEL=%PAPER_DIR%\src\fontdiffuser\configs\model.yaml
set TRAIN=%PAPER_DIR%\src\fontdiffuser\configs\train_stage_b_long.yaml
set INIT=%PAPER_DIR%\outputs\stage_b\fontdiffuser_last.pt
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\01_stage_b_long_gpu1_%DT%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d %PAPER_DIR%
uv run python -u -m paper_reimpl_shared.runner.entrypoint --paper fontdiffuser --train "%TRAIN%" --model "%MODEL%" --data "%DATA%" --data-backend lab_server --device cuda:1 --init-ckpt "%INIT%" > "%LOG%" 2>&1
endlocal
