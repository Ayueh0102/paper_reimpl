@echo off
chcp 65001 >nul
setlocal
REM 02_hfh_font VAE pretrain on cuda:1 (parallel to 01 v3 on cuda:0).
REM Trains TinyVAE on glyph reconstruction loss. Output ckpt is loaded
REM by the regular Stage A run via --init-ckpt + freeze_vae:true.

set REPO=D:\Char\ayueh\paper_reimpl\repo
set PAPER_DIR=%REPO%\papers\02_hfh_font
set FONTS=D:\Char\ayueh\paper_reimpl\data_snapshot\fonts_free
set OUT=%PAPER_DIR%\outputs\stage_vae\vae_last.pt

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\02_vae_pretrain_%DT%.log

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

cd /d %PAPER_DIR%
C:\Users\Ptri\.local\bin\uv.exe run python scripts\pretrain_vae.py ^
    --fonts-root "%FONTS%" --output "%OUT%" ^
    --steps 5000 --batch-size 32 --lr 2e-4 --kl-weight 1e-6 ^
    --log-every 100 --device cuda:1 ^
    > "%LOG%" 2>&1

endlocal
