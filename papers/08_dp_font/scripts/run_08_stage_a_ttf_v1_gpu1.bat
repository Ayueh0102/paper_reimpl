@echo off
chcp 65001 >nul
setlocal
set REPO=D:\Char\ayueh\paper_reimpl\repo
set PAPER_DIR=%REPO%\papers\08_dp_font
set DATA=%PAPER_DIR%\src\dp_font\configs\data_stage_a_ttf.yaml
set MODEL=%PAPER_DIR%\src\dp_font\configs\model.yaml
set TRAIN=%PAPER_DIR%\src\dp_font\configs\train_stage_a_ttf_v1.yaml

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\08_stage_a_ttf_v1_%DT%.log

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

cd /d %PAPER_DIR%
C:\Users\Ptri\.local\bin\uv.exe run python -m paper_reimpl_shared.runner.entrypoint ^
    --paper dp_font ^
    --train "%TRAIN%" --model "%MODEL%" --data "%DATA%" ^
    --data-backend lab_server --device cuda:1 ^
    > "%LOG%" 2>&1

endlocal
