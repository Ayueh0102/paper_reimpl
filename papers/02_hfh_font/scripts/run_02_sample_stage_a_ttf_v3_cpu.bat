@echo off
chcp 65001 >nul
setlocal
REM 02 Stage A v3 (paper-faithful: frozen pretrained VAE) sample on CPU.

set REPO=D:\Char\ayueh\paper_reimpl\repo
set PAPER_DIR=%REPO%\papers\02_hfh_font
set CKPT=%PAPER_DIR%\outputs\stage_a_ttf_v3\hfh_font_last.pt
set FONTS=D:\Char\ayueh\paper_reimpl\data_snapshot\fonts_free
set OUT=%PAPER_DIR%\outputs\stage_a_ttf_v3\sample_grid.png

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\02_sample_stage_a_ttf_v3_cpu_%DT%.log

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

cd /d %PAPER_DIR%
uv run python -u scripts\sample_stage_a_ttf_v1.py ^
    --ckpt "%CKPT%" --fonts-root "%FONTS%" --output "%OUT%" ^
    --n 12 --ddim-steps 50 --cfg-scale 2.0 --device cpu ^
    > "%LOG%" 2>&1

endlocal
