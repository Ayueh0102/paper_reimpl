@echo off
chcp 65001 >nul
setlocal
REM 03 IF-Font Stage A v2 sample on CPU (cuda left alone).
set REPO=D:\Char\ayueh\paper_reimpl\repo
set PAPER_DIR=%REPO%\papers\03_if_font
set FONTS=D:\Char\ayueh\paper_reimpl\data_snapshot\fonts_free
set CKPT=%PAPER_DIR%\outputs\stage_a_ttf_v2\if_font_last.pt
set OUT=%PAPER_DIR%\outputs\stage_a_ttf_v2\sample_grid.png
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\03_sample_v2_cpu_%DT%.log

set PYTHONUNBUFFERED=1
cd /d %PAPER_DIR%
uv run python -u scripts/sample_stage_a_ttf_v2.py ^
    --ckpt "%CKPT%" ^
    --fonts-root "%FONTS%" ^
    --output "%OUT%" ^
    --n 12 --device cpu --seed 2026 ^
    --image-size 128 --max-refs 3 > "%LOG%" 2>&1
endlocal
