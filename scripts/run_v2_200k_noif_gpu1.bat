@echo off
chcp 65001 >nul
setlocal
REM v2 200k chain on cuda:1 — SKIP 03 (waiting for VQGAN pretrain).
REM Order: 06 Calliffusion (heaviest) → 04 VQ-Font → 05 QT-Font → 08 DP-Font.

set REPO=D:\Char\ayueh\paper_reimpl\repo
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set CHAIN_DT=%%I
set CHAIN_LOG=D:\Char\ayueh\paper_reimpl\logs\v2_200k_noif_gpu1_%CHAIN_DT%.log

echo === v2 200k no-IF chain on cuda:1 started %DATE% %TIME% === > "%CHAIN_LOG%"

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
echo === v2 200k no-IF chain done %DATE% %TIME% === >> "%CHAIN_LOG%"
endlocal
