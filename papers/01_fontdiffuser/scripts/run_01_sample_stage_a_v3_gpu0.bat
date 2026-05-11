@echo off
chcp 65001 >nul
setlocal
REM 12-cell sample grid for Stage A v3 ckpt (80k step).
REM Same script as v1 sampler; just points at the v3 ckpt + output dir.

set REPO=D:\Char\ayueh\paper_reimpl\repo
set PAPER_DIR=%REPO%\papers\01_fontdiffuser
set CKPT=%PAPER_DIR%\outputs\stage_a_ttf_v3\fontdiffuser_last.pt
set FONTS=D:\Char\ayueh\paper_reimpl\data_snapshot\fonts_free
set OUT=%PAPER_DIR%\outputs\stage_a_ttf_v3\sample_grid.png

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\01_sample_stage_a_v3_%DT%.log

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

cd /d %PAPER_DIR%
C:\Users\Ptri\.local\bin\uv.exe run python scripts\sample_stage_a_v1.py ^
    --ckpt "%CKPT%" --fonts-root "%FONTS%" --output "%OUT%" ^
    --n 12 --ddim-steps 50 --cfg-scale 1.0 --device cuda:0 ^
    > "%LOG%" 2>&1

endlocal
