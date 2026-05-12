@echo off
chcp 65001 >nul
setlocal
REM 03 IF-Font Stage-1 VQGAN pretrain on cuda:0.
REM Trains _StubVQGANEncoder + _StubVQGANDecoder + 256-entry codebook on
REM 13 OFL TTF fonts (~30k step, bs=32). Output is a frozen tokenizer for
REM Phase-2 Stage A v2 training.

set REPO=D:\Char\ayueh\paper_reimpl\repo
set FONTS=D:\Char\ayueh\paper_reimpl\data_snapshot\fonts_free
set OUT=%REPO%\papers\03_if_font\outputs\stage_vqgan\vqgan_last.pt
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\03_pretrain_vqgan_%DT%.log

cd /d %REPO%\papers\03_if_font
set PYTHONUNBUFFERED=1
uv run python -u scripts/pretrain_vqgan.py ^
    --fonts-root "%FONTS%" ^
    --output "%OUT%" ^
    --steps 30000 ^
    --batch-size 32 ^
    --lr 2e-4 ^
    --commit-weight 0.25 ^
    --log-every 200 ^
    --device cuda:0 ^
    --image-size 128 > "%LOG%" 2>&1
endlocal
