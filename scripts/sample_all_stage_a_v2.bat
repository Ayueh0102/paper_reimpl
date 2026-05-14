@echo off
chcp 65001 >nul
setlocal
REM Sample all 4 paper Stage A v2 ckpts in sequence on cuda:0 (cuda:1 idle too).
REM Order: 04 -> 06 -> 08 -> 05. Each writes outputs/stage_a_ttf_v2/sample_grid.png.

set REPO=D:\Char\ayueh\paper_reimpl\repo
set FONTS=D:\Char\ayueh\paper_reimpl\data_snapshot\fonts_free
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\sample_all_v2_%DT%.log

echo === sample_all_v2 started %DATE% %TIME% === > "%LOG%"

echo --- 04 vq_font --- >> "%LOG%"
cd /d %REPO%\papers\04_vq_font
uv run python -u scripts/sample_stage_a_ttf_v2.py ^
    --ckpt outputs/stage_a_ttf_v2/transformer_last.pt ^
    --fonts-root "%FONTS%" ^
    --output outputs/stage_a_ttf_v2/sample_grid.png ^
    --n 12 --device cuda:0 --image-size 128 >> "%LOG%" 2>&1

echo --- 06 calliffusion --- >> "%LOG%"
cd /d %REPO%\papers\06_calliffusion
uv run python -u scripts/sample_stage_a_ttf_v2.py ^
    --ckpt outputs/stage_a_ttf_v2/calliffusion_last.pt ^
    --fonts-root "%FONTS%" ^
    --output outputs/stage_a_ttf_v2/sample_grid.png ^
    --n 12 --device cuda:0 --image-size 64 --cfg-scale 2.0 >> "%LOG%" 2>&1

echo --- 08 dp_font --- >> "%LOG%"
cd /d %REPO%\papers\08_dp_font
uv run python -u scripts/sample_stage_a_ttf_v2.py ^
    --ckpt outputs/stage_a_ttf_v2/dp_font_last.pt ^
    --fonts-root "%FONTS%" ^
    --output outputs/stage_a_ttf_v2/sample_grid.png ^
    --n 12 --device cuda:0 --image-size 80 --cfg-scale 2.0 >> "%LOG%" 2>&1

echo --- 05 qt_font (cuda:1) --- >> "%LOG%"
cd /d %REPO%\papers\05_qt_font
uv run python -u scripts/sample_stage_a_ttf_v2.py ^
    --ckpt outputs/stage_a_ttf_v2/qt_font_last.pt ^
    --fonts-root "%FONTS%" ^
    --output outputs/stage_a_ttf_v2/sample_grid.png ^
    --n 12 --device cuda:1 >> "%LOG%" 2>&1

echo === sample_all_v2 done %DATE% %TIME% === >> "%LOG%"
endlocal
