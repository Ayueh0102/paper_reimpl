@echo off
chcp 65001 >nul
setlocal
REM Stage A v1 — real TTF cross-font pretrain (5000 steps) on cuda:0.
REM First-run note: TTFCrossFontPairDataset discovers the shared CJK char
REM set by rendering ~20k probe glyphs per font. First start takes ~4 min;
REM subsequent starts hit the cached JSON (sub-second).

set REPO=D:\Char\ayueh\paper_reimpl\repo
set PAPER_DIR=%REPO%\papers\01_fontdiffuser
set DATA=%PAPER_DIR%\src\fontdiffuser\configs\data_stage_a_ttf.yaml
set MODEL=%PAPER_DIR%\src\fontdiffuser\configs\model.yaml
set TRAIN=%PAPER_DIR%\src\fontdiffuser\configs\train_stage_a_ttf_v1.yaml

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\01_stage_a_ttf_v1_%DT%.log

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

cd /d %PAPER_DIR%
C:\Users\Ptri\.local\bin\uv.exe run python -m paper_reimpl_shared.runner.entrypoint ^
    --paper fontdiffuser ^
    --train "%TRAIN%" --model "%MODEL%" --data "%DATA%" ^
    --data-backend lab_server --device cuda:0 ^
    > "%LOG%" 2>&1

endlocal
