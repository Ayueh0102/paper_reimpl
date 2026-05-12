@echo off
chcp 65001 >nul
setlocal
REM 02 VAE pretrain v2 — 30000 step, 6x longer than v1.
REM Output: outputs/stage_vae_v2/vae_v2_last.pt for Stage A v3 warm-start.

set REPO=D:\Char\ayueh\paper_reimpl\repo
set PAPER_DIR=%REPO%\papers\02_hfh_font
set FONTS=D:\Char\ayueh\paper_reimpl\data_snapshot\fonts_free
set OUT=%PAPER_DIR%\outputs\stage_vae_v2\vae_v2_last.pt

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\02_vae_pretrain_v2_%DT%.log

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

cd /d %PAPER_DIR%
C:\Users\Ptri\.local\bin\uv.exe run python scripts\pretrain_vae.py ^
    --fonts-root "%FONTS%" --output "%OUT%" ^
    --steps 30000 --batch-size 32 --lr 2e-4 --kl-weight 1e-6 ^
    --log-every 500 --device cuda:1 ^
    > "%LOG%" 2>&1

endlocal
