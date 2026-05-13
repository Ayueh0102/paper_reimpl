@echo off
chcp 65001 >/dev/null
setlocal
REM 05_qt_font Stage A v2 — 30k step at depth=6 (image_size=64) from scratch.
REM v1 ckpt was at depth=7 so it can't warm-start the new depth=6 model.

set REPO=D:\Char\ayueh\paper_reimpl\repo
set PAPER_DIR=%REPO%\papers\05_qt_font
set DATA=%PAPER_DIR%\src\qt_font\configs\data_stage_a_ttf.yaml
set MODEL=%PAPER_DIR%\src\qt_font\configs\model.yaml
set TRAIN=%PAPER_DIR%\src\qt_font\configs\train_stage_a_ttf_v2.yaml

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\05_stage_a_ttf_v2_%DT%.log

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

cd /d %PAPER_DIR%
uv run python -u -m paper_reimpl_shared.runner.entrypoint ^
    --paper qt_font ^
    --train "%TRAIN%" --model "%MODEL%" --data "%DATA%" ^
    --data-backend lab_server --device cuda:1 ^
    > "%LOG%" 2>&1

endlocal
