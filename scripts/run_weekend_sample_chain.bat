@echo off
chcp 65001 >nul
setlocal
REM Sunday-morning sample chain — re-sample 08/01/02/04/05 from their SB-long ckpts.
set REPO=D:\Char\ayueh\paper_reimpl\repo
set MANIFEST=D:\Char\ayueh\paper_reimpl\data_snapshot\splits\a_main_clean_split_character_disjoint_global_coverage_enriched.jsonl
set FONTS=D:\Char\ayueh\paper_reimpl\data_snapshot\fonts_free
set LOG=D:\Char\ayueh\paper_reimpl\logs\weekend_sample_chain.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo === 08 SB extra-long 200k ernantang sample === > "%LOG%"
cd /d %REPO%\papers\08_dp_font
uv run python -u scripts\sample_stage_b_ernantang.py ^
  --ckpt outputs\stage_b_extra_long\dp_font_last.pt ^
  --manifest "%MANIFEST%" --fonts-root "%FONTS%" ^
  --output outputs\stage_b_extra_long\sample_grid_ernantang.png ^
  --n 12 --device cuda:1 --image-size 80 --cfg-scale 2.0 >> "%LOG%" 2>&1

echo === 01 FontDiffuser SB-long 50k ernantang sample === >> "%LOG%"
cd /d %REPO%\papers\01_fontdiffuser
uv run python -u scripts\sample_stage_b_ernantang.py ^
  --ckpt outputs\stage_b_long\fontdiffuser_last.pt ^
  --manifest "%MANIFEST%" --fonts-root "%FONTS%" ^
  --output outputs\stage_b_long\sample_grid_ernantang.png ^
  --n 12 --device cuda:1 --cfg-scale 1.0 >> "%LOG%" 2>&1

echo === 02 HFH-Font SB-long 30k ernantang sample === >> "%LOG%"
cd /d %REPO%\papers\02_hfh_font
uv run python -u scripts\sample_stage_b_ernantang.py ^
  --ckpt outputs\stage_b_long\hfh_font_last.pt ^
  --manifest "%MANIFEST%" --fonts-root "%FONTS%" ^
  --output outputs\stage_b_long\sample_grid_ernantang.png ^
  --n 12 --device cuda:1 --cfg-scale 2.0 >> "%LOG%" 2>&1

echo === 04 VQ-Font SB-long 30k ernantang sample === >> "%LOG%"
cd /d %REPO%\papers\04_vq_font
uv run python -u scripts\sample_stage_b_ernantang.py ^
  --ckpt outputs\stage_b_long\transformer_last.pt ^
  --manifest "%MANIFEST%" --fonts-root "%FONTS%" ^
  --output outputs\stage_b_long\sample_grid_ernantang.png ^
  --n 12 --device cuda:1 >> "%LOG%" 2>&1

echo === 05 QT-Font SB ernantang sample === >> "%LOG%"
cd /d %REPO%\papers\05_qt_font
uv run python -u scripts\sample_stage_b_ernantang.py ^
  --ckpt outputs\stage_b\qt_font_last.pt ^
  --manifest "%MANIFEST%" --fonts-root "%FONTS%" ^
  --output outputs\stage_b\sample_grid_ernantang.png ^
  --n 12 --device cuda:1 --ddim-steps 50 >> "%LOG%" 2>&1

echo === sample chain done === >> "%LOG%"
endlocal
