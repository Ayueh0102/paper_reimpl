@echo off
chcp 65001 >nul
setlocal
REM 06 Stage A v1 — BERT path may download bert-base-chinese (~1.3GB) on
REM first run. Make sure D: has space and HF cache is writable.

set REPO=D:\Char\ayueh\paper_reimpl\repo
set PAPER_DIR=%REPO%\papers\06_calliffusion
set DATA=%PAPER_DIR%\src\calliffusion\configs\data_stage_a_ttf.yaml
set MODEL=%PAPER_DIR%\src\calliffusion\configs\model.yaml
set TRAIN=%PAPER_DIR%\src\calliffusion\configs\train_stage_a_ttf_v1.yaml

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\06_stage_a_ttf_v1_%DT%.log

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

cd /d %PAPER_DIR%
C:\Users\Ptri\.local\bin\uv.exe run python -m paper_reimpl_shared.runner.entrypoint ^
    --paper calliffusion ^
    --train "%TRAIN%" --model "%MODEL%" --data "%DATA%" ^
    --data-backend lab_server --device cuda:0 ^
    > "%LOG%" 2>&1

endlocal
