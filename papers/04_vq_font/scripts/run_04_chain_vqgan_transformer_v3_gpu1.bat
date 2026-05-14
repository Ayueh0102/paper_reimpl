@echo off
chcp 65001 >nul
setlocal
REM 04 v3 chain — first Stage 0 VQGAN 50k, then Stage Transformer 200k
REM warm-starting from the v3 VQGAN ckpt.

set REPO=D:\Char\ayueh\paper_reimpl\repo
set PAPER_DIR=%REPO%\papers\04_vq_font
set DATA_VQGAN=%PAPER_DIR%\src\vq_font\configs\data_stage_0_vqgan.yaml
set DATA_TFM=%PAPER_DIR%\src\vq_font\configs\data_stage_a_ttf.yaml
set MODEL=%PAPER_DIR%\src\vq_font\configs\model.yaml
set TRAIN_VQGAN=%PAPER_DIR%\src\vq_font\configs\train_stage_0_vqgan_v3.yaml
set TRAIN_TFM=%PAPER_DIR%\src\vq_font\configs\train_stage_a_ttf_v3.yaml

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\04_chain_v3_%DT%.log
set LOG_VQGAN=D:\Char\ayueh\paper_reimpl\logs\04_stage_0_vqgan_v3_%DT%.log
set LOG_TFM=D:\Char\ayueh\paper_reimpl\logs\04_stage_a_ttf_v3_%DT%.log

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo === 04 v3 chain started %DATE% %TIME% === > "%LOG%"

echo --- Stage 0 VQGAN 50k --- >> "%LOG%"
cd /d %PAPER_DIR%
uv run python -u -m paper_reimpl_shared.runner.entrypoint ^
    --paper vq_font ^
    --train "%TRAIN_VQGAN%" --model "%MODEL%" --data "%DATA_VQGAN%" ^
    --data-backend lab_server --device cuda:1 ^
    > "%LOG_VQGAN%" 2>&1

echo VQGAN done errorlevel=%errorlevel% >> "%LOG%"

echo --- Stage Transformer 200k --- >> "%LOG%"
uv run python -u -m paper_reimpl_shared.runner.entrypoint ^
    --paper vq_font ^
    --train "%TRAIN_TFM%" --model "%MODEL%" --data "%DATA_TFM%" ^
    --data-backend lab_server --device cuda:1 ^
    > "%LOG_TFM%" 2>&1

echo Transformer done errorlevel=%errorlevel% >> "%LOG%"
echo === 04 v3 chain done %DATE% %TIME% === >> "%LOG%"
endlocal
