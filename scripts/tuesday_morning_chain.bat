@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
REM Tuesday morning chain — fires after 02 SB-long completes Mon overnight.
REM Sequentially:
REM   1. Re-train 05 SB 5k on cuda:1 (~30 min, gives us a fresh 05 ckpt with logging fix)
REM   2. Sample 04 SB-long 70k on cuda:1 (~5 min)
REM   3. Sample 02 SB-long 80k on cuda:1 (~5 min)
REM   4. Re-sample 01 SB-long 100k on cuda:1 (~5 min, replaces the CPU version)
REM   5. Sample 05 SB 5k fresh on cuda:1 (~3 min)
REM
REM Trigger condition: wait for 02 SB-long ckpt to exist.

set LOG_DIR=D:\Char\ayueh\paper_reimpl\logs
set REPO=D:\Char\ayueh\paper_reimpl\repo
set MANIFEST=D:\Char\ayueh\paper_reimpl\data_snapshot\splits\a_main_clean_split_character_disjoint_global_coverage_enriched.jsonl
set FONTS=D:\Char\ayueh\paper_reimpl\data_snapshot\fonts_free
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set SUP=%LOG_DIR%\tuesday_morning_chain_%DT%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
echo === tuesday morning chain started %DATE% %TIME% === > "%SUP%"

REM Wait for 02 SB-long to complete (or 02 ckpt to exist).
echo --- waiting for 02 SB-long ckpt --- >> "%SUP%"
:wait_02
timeout /t 180 /nobreak >nul
if exist %REPO%\papers\02_hfh_font\outputs\stage_b_long\hfh_font_last.pt goto have_02
powershell -NoProfile -Command "$f = Get-ChildItem '%LOG_DIR%\02_stage_b_long_gpu1_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f -and (Get-Content $f.FullName -Raw 2>$null) -match 'Traceback') { exit 0 } else { exit 1 }"
if errorlevel 1 goto wait_02
:have_02
echo %DATE% %TIME% 02 SB-long ready >> "%SUP%"

REM Phase 1: re-train 05 SB 5k (logging fix applied earlier).
echo --- phase 1: 05 SB 5k re-train --- >> "%SUP%"
cd /d %REPO%\papers\05_qt_font
set TRAIN05=%REPO%\papers\05_qt_font\src\qt_font\configs\train_stage_b_midtrain.yaml
set MODEL05=%REPO%\papers\05_qt_font\src\qt_font\configs\model.yaml
set DATA05=%REPO%\papers\05_qt_font\src\qt_font\configs\data_stage_b.yaml
set INIT05=%REPO%\papers\05_qt_font\outputs\stage_a_ttf_v2\qt_font_last.pt
set LOG05=%LOG_DIR%\05_stage_b_retrain_gpu1_%DT%.log
uv run python -u -m paper_reimpl_shared.runner.entrypoint --paper qt_font --train "%TRAIN05%" --model "%MODEL05%" --data "%DATA05%" --data-backend lab_server --device cuda:1 --init-ckpt "%INIT05%" > "%LOG05%" 2>&1
echo %DATE% %TIME% 05 SB retrain done >> "%SUP%"

REM Phase 2: sample 04 SB-long 70k
echo --- phase 2: sample 04 SB-long --- >> "%SUP%"
cd /d %REPO%\papers\04_vq_font
uv run python -u scripts\sample_stage_b_ernantang.py ^
  --ckpt outputs\stage_b_long\transformer_last.pt ^
  --manifest "%MANIFEST%" --fonts-root "%FONTS%" ^
  --output outputs\stage_b_long\sample_grid_ernantang.png ^
  --n 12 --device cuda:1 >> "%SUP%" 2>&1

REM Phase 3: sample 02 SB-long 80k
echo --- phase 3: sample 02 SB-long --- >> "%SUP%"
cd /d %REPO%\papers\02_hfh_font
uv run python -u scripts\sample_stage_b_ernantang.py ^
  --ckpt outputs\stage_b_long\hfh_font_last.pt ^
  --manifest "%MANIFEST%" --fonts-root "%FONTS%" ^
  --output outputs\stage_b_long\sample_grid_ernantang.png ^
  --n 12 --device cuda:1 --cfg-scale 2.0 >> "%SUP%" 2>&1

REM Phase 4: re-sample 01 SB-long 100k on GPU
echo --- phase 4: re-sample 01 SB-long on GPU --- >> "%SUP%"
cd /d %REPO%\papers\01_fontdiffuser
uv run python -u scripts\sample_stage_b_ernantang.py ^
  --ckpt outputs\stage_b_long\fontdiffuser_last.pt ^
  --manifest "%MANIFEST%" --fonts-root "%FONTS%" ^
  --output outputs\stage_b_long\sample_grid_ernantang.png ^
  --n 12 --device cuda:1 --cfg-scale 1.0 >> "%SUP%" 2>&1

REM Phase 5: sample 05 freshly trained
echo --- phase 5: sample 05 SB --- >> "%SUP%"
cd /d %REPO%\papers\05_qt_font
uv run python -u scripts\sample_stage_b_ernantang.py ^
  --ckpt outputs\stage_b\qt_font_last.pt ^
  --manifest "%MANIFEST%" --fonts-root "%FONTS%" ^
  --output outputs\stage_b\sample_grid_ernantang.png ^
  --n 12 --device cuda:1 --ddim-steps 50 >> "%SUP%" 2>&1

echo === tuesday morning chain done %DATE% %TIME% === >> "%SUP%"
endlocal
