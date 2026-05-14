@echo off
chcp 65001 >nul
setlocal
REM 08 Stage B PINN OFF ablation (pinn_weight=0).
set REPO=D:\Char\ayueh\paper_reimpl\repo
set PAPER_DIR=%REPO%\papers\08_dp_font
set DATA=%PAPER_DIR%\src\dp_font\configs\data_stage_b.yaml
set MODEL=%PAPER_DIR%\src\dp_font\configs\model.yaml
set TRAIN=%PAPER_DIR%\src\dp_font\configs\train_stage_b_midtrain_pinn0.yaml
set INIT=%PAPER_DIR%\outputs\stage_a_ttf_v3\dp_font_last.pt
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\08_stage_b_pinn0_gpu0_%DT%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d %PAPER_DIR%
uv run python -u -m paper_reimpl_shared.runner.entrypoint ^
    --paper dp_font ^
    --train "%TRAIN%" --model "%MODEL%" --data "%DATA%" ^
    --data-backend lab_server --device cuda:0 ^
    --init-ckpt "%INIT%" > "%LOG%" 2>&1
endlocal
