@echo off
chcp 65001 >nul
setlocal
REM 200k Stage A v2 chain on cuda:0 — heavier papers (03 IF-Font, 04 VQ-Font, 06 Calliffusion).
REM Each paper warm-starts from its own v1 ckpt. ~9-11 hr per paper × 3 = ~27-33 hr wall.

set REPO=D:\Char\ayueh\paper_reimpl\repo
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set CHAIN_DT=%%I
set CHAIN_LOG=D:\Char\ayueh\paper_reimpl\logs\v2_200k_chain_gpu0_%CHAIN_DT%.log

echo === v2 200k chain gpu0 started %DATE% %TIME% === > "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === 03_if_font Stage A v2 (200k) %DATE% %TIME% >> "%CHAIN_LOG%"
call "%REPO%\papers\03_if_font\scripts\run_03_stage_a_ttf_v2_gpu0.bat"
echo === 03_if_font finished errorlevel=%errorlevel% %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === 04_vq_font Stage A v2 (200k) %DATE% %TIME% >> "%CHAIN_LOG%"
call "%REPO%\papers\04_vq_font\scripts\run_04_stage_a_ttf_v2_gpu0.bat"
echo === 04_vq_font finished errorlevel=%errorlevel% %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === 06_calliffusion Stage A v2 (200k) %DATE% %TIME% >> "%CHAIN_LOG%"
call "%REPO%\papers\06_calliffusion\scripts\run_06_stage_a_ttf_v2_gpu0.bat"
echo === 06_calliffusion finished errorlevel=%errorlevel% %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === v2 200k chain gpu0 done %DATE% %TIME% === >> "%CHAIN_LOG%"
endlocal
