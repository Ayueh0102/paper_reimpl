@echo off
chcp 65001 >nul
setlocal
REM Phase 3 shakedown: synthetic Stage A on cuda:0 (200 steps).
REM Validates the full GPU pipeline (CUDA, DataLoader, model fwd/bwd, ckpt save)
REM before we invest in writing the real TTF cross-font dataset.

set REPO=D:\Char\ayueh\paper_reimpl\repo
set PAPER_DIR=%REPO%\papers\01_fontdiffuser
set DATA=%PAPER_DIR%\src\fontdiffuser\configs\data_stage_a.yaml
set MODEL=%PAPER_DIR%\src\fontdiffuser\configs\model.yaml
set TRAIN=%PAPER_DIR%\src\fontdiffuser\configs\train_stage_a_shakedown.yaml

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\01_stage_a_shakedown_%DT%.log

set PYTHONIOENCODING=utf-8

cd /d %PAPER_DIR%
C:\Users\Ptri\.local\bin\uv.exe run python -m paper_reimpl_shared.runner.entrypoint ^
    --paper fontdiffuser ^
    --train "%TRAIN%" --model "%MODEL%" --data "%DATA%" ^
    --data-backend lab_server --device cuda:0 ^
    > "%LOG%" 2>&1

endlocal
