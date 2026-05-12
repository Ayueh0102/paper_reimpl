@echo off
chcp 65001 >nul
setlocal
REM 200k Stage A v2 chain on cuda:1 — lighter papers (05 QT-Font, 08 DP-Font).
REM Each paper warm-starts from its own v1 ckpt. ~9 hr per paper × 2 = ~18 hr wall.

set REPO=D:\Char\ayueh\paper_reimpl\repo
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set CHAIN_DT=%%I
set CHAIN_LOG=D:\Char\ayueh\paper_reimpl\logs\v2_200k_chain_gpu1_%CHAIN_DT%.log

echo === v2 200k chain gpu1 started %DATE% %TIME% === > "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === 05_qt_font Stage A v2 (200k) %DATE% %TIME% >> "%CHAIN_LOG%"
call "%REPO%\papers\05_qt_font\scripts\run_05_stage_a_ttf_v2_gpu1.bat"
echo === 05_qt_font finished errorlevel=%errorlevel% %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === 08_dp_font Stage A v2 (200k) %DATE% %TIME% >> "%CHAIN_LOG%"
call "%REPO%\papers\08_dp_font\scripts\run_08_stage_a_ttf_v2_gpu1.bat"
echo === 08_dp_font finished errorlevel=%errorlevel% %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === v2 200k chain gpu1 done %DATE% %TIME% === >> "%CHAIN_LOG%"
endlocal
