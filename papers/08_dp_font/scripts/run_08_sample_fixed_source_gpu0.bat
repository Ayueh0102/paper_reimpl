@echo off
chcp 65001 >nul
setlocal
REM 08 sample with fixed source = noto_sans_sc, varying writer_id.
REM Tests content-copy hypothesis.

set REPO=D:\Char\ayueh\paper_reimpl\repo
set PAPER_DIR=%REPO%\papers\08_dp_font
set CKPT=%PAPER_DIR%\outputs\stage_a_ttf_v2\dp_font_last.pt
set FONTS=D:\Char\ayueh\paper_reimpl\data_snapshot\fonts_free
set OUT=%PAPER_DIR%\outputs\stage_a_ttf_v2\sample_grid_fixed_source.png

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\08_sample_fixed_source_%DT%.log

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

cd /d %PAPER_DIR%
uv run python -u scripts\sample_08_fixed_source.py ^
    --ckpt "%CKPT%" --fonts-root "%FONTS%" --output "%OUT%" ^
    --n 12 --device cuda:0 --source-font noto_sans_sc --cfg-scale 2.0 ^
    > "%LOG%" 2>&1

endlocal
