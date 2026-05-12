@echo off
chcp 65001 >/dev/null
setlocal
REM 06_calliffusion Stage A v2 — 200k step warm-start from v1 ckpt (cuda:1).

set REPO=D:\Char\ayueh\paper_reimpl\repo
set PAPER_DIR=%REPO%\papers\06_calliffusion
set DATA=%PAPER_DIR%\src\calliffusion\configs\data_stage_a_ttf.yaml
set MODEL=%PAPER_DIR%\src\calliffusion\configs\model.yaml
set TRAIN=%PAPER_DIR%\src\calliffusion\configs\train_stage_a_ttf_v2.yaml
set INIT=%PAPER_DIR%\outputs\stage_a_ttf_v1\calliffusion_last.pt

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\06_stage_a_ttf_v2_%DT%.log

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

cd /d %PAPER_DIR%
C:\Users\Ptri\.local\bin\uv.exe run python -m paper_reimpl_shared.runner.entrypoint ^
    --paper calliffusion ^
    --train "%TRAIN%" --model "%MODEL%" --data "%DATA%" ^
    --data-backend lab_server --device cuda:1 ^
    --init-ckpt "%INIT%" ^
    > "%LOG%" 2>&1

endlocal
