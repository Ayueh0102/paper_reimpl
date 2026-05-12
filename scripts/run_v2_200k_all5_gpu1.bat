@echo off
chcp 65001 >nul
setlocal
REM v2 200k chain — ALL 5 papers on cuda:1, ordered 03/06/04/05/08.
REM Heavier models first (03 IF-Font, 06 Calliffusion). 04 VQ-Font and
REM the smaller 05/08 trail. Total ~45-55 hr wall on a single GPU.

set REPO=D:\Char\ayueh\paper_reimpl\repo
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set CHAIN_DT=%%I
set CHAIN_LOG=D:\Char\ayueh\paper_reimpl\logs\v2_200k_all5_gpu1_%CHAIN_DT%.log

echo === v2 200k all-5 chain on cuda:1 started %DATE% %TIME% === > "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === 03_if_font Stage A v2 (200k) %DATE% %TIME% >> "%CHAIN_LOG%"
call "%REPO%\papers\03_if_font\scripts\run_03_stage_a_ttf_v2_gpu1.bat"
echo === 03_if_font finished errorlevel=%errorlevel% %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === 06_calliffusion Stage A v2 (200k) %DATE% %TIME% >> "%CHAIN_LOG%"
call "%REPO%\papers\06_calliffusion\scripts\run_06_stage_a_ttf_v2_gpu1.bat"
echo === 06_calliffusion finished errorlevel=%errorlevel% %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === 04_vq_font Stage A v2 (200k) %DATE% %TIME% >> "%CHAIN_LOG%"
call "%REPO%\papers\04_vq_font\scripts\run_04_stage_a_ttf_v2_gpu1.bat"
echo === 04_vq_font finished errorlevel=%errorlevel% %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === 05_qt_font Stage A v2 (200k) %DATE% %TIME% >> "%CHAIN_LOG%"
call "%REPO%\papers\05_qt_font\scripts\run_05_stage_a_ttf_v2_gpu1.bat"
echo === 05_qt_font finished errorlevel=%errorlevel% %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === 08_dp_font Stage A v2 (200k) %DATE% %TIME% >> "%CHAIN_LOG%"
call "%REPO%\papers\08_dp_font\scripts\run_08_stage_a_ttf_v2_gpu1.bat"
echo === 08_dp_font finished errorlevel=%errorlevel% %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === v2 200k all-5 chain done %DATE% %TIME% === >> "%CHAIN_LOG%"
endlocal
