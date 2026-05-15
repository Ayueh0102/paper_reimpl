@echo off
chcp 65001 >nul
setlocal
REM 08 Stage B sample on REAL ernantang writers, CPU (cuda:1 + cuda:0 busy).
set REPO=D:\Char\ayueh\paper_reimpl\repo
set PAPER_DIR=%REPO%\papers\08_dp_font
set CKPT=%PAPER_DIR%\outputs\stage_b\dp_font_last.pt
set MANIFEST=D:\Char\ayueh\paper_reimpl\data_snapshot\splits\a_main_clean_split_character_disjoint_global_coverage_enriched.jsonl
set FONTS=D:\Char\ayueh\paper_reimpl\data_snapshot\fonts_free
set OUT=%PAPER_DIR%\outputs\stage_b\sample_grid_ernantang.png

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\08_sample_stage_b_ernantang_cpu_%DT%.log

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

cd /d %PAPER_DIR%
uv run python -u scripts\sample_stage_b_ernantang.py ^
    --ckpt "%CKPT%" --manifest "%MANIFEST%" --fonts-root "%FONTS%" --output "%OUT%" ^
    --n 12 --device cpu --image-size 80 --cfg-scale 2.0 ^
    > "%LOG%" 2>&1

endlocal
