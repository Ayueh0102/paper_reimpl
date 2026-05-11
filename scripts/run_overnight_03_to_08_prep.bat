@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
REM Overnight chain — for each of papers 03/04/05/06/07/08:
REM   1. uv sync (rebuild venv from uv.lock with cu128 torch)
REM   2. pytest tests/test_smoke.py -x
REM Each step's full output is appended to a single log; failures are
REM noted but the chain continues to the next paper so we get a complete
REM audit of which papers actually install + smoke on lab Windows.

set REPO=D:\Char\ayueh\paper_reimpl\repo
set UV=C:\Users\Ptri\.local\bin\uv.exe
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set LOG=D:\Char\ayueh\paper_reimpl\logs\overnight_03to08_prep_%DT%.log

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo === overnight 03-08 prep started %DATE% %TIME% === > "%LOG%"

for %%P in (03_if_font 04_vq_font 05_qt_font 06_calliffusion 07_moyun 08_dp_font) do (
    echo. >> "%LOG%"
    echo ============================================== >> "%LOG%"
    echo === %%P : uv sync   %DATE% %TIME% >> "%LOG%"
    echo ============================================== >> "%LOG%"
    cd /d %REPO%\papers\%%P
    %UV% sync >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo === %%P : uv sync FAILED, skipping smoke >> "%LOG%"
    ) else (
        echo === %%P : uv sync OK >> "%LOG%"
        echo === %%P : pytest smoke   %DATE% %TIME% >> "%LOG%"
        %UV% run pytest tests/test_smoke.py -x -v >> "%LOG%" 2>&1
        if errorlevel 1 (
            echo === %%P : smoke FAILED >> "%LOG%"
        ) else (
            echo === %%P : smoke OK >> "%LOG%"
        )
    )
)

echo. >> "%LOG%"
echo === overnight 03-08 prep finished %DATE% %TIME% === >> "%LOG%"
endlocal
