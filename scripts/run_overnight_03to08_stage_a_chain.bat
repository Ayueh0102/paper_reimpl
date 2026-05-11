@echo off
chcp 65001 >nul
setlocal
REM Overnight chain — sequential Stage A v1 (5000 step TTF) for papers
REM 03/04/05/06/08 on cuda:0. 07_moyun excluded (latent-diffusion, needs
REM VAE pretrain first, same gap as 02). Each paper's training logs to
REM its own timestamped file via its own bat; this chain log records
REM only the chain-level errorlevels + timings.
REM
REM Estimated wall time: 5 papers × ~20 min ≈ 100 min total.

set REPO=D:\Char\ayueh\paper_reimpl\repo
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set CHAIN_DT=%%I
set CHAIN_LOG=D:\Char\ayueh\paper_reimpl\logs\overnight_03to08_stage_a_chain_%CHAIN_DT%.log

echo === overnight 03-08 stage A chain started %DATE% %TIME% === > "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === 03_if_font Stage A TTF v1   %DATE% %TIME% >> "%CHAIN_LOG%"
call "%REPO%\papers\03_if_font\scripts\run_03_stage_a_ttf_v1_gpu0.bat"
echo === 03_if_font finished errorlevel=%errorlevel%   %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === 04_vq_font Stage A TTF v1   %DATE% %TIME% >> "%CHAIN_LOG%"
call "%REPO%\papers\04_vq_font\scripts\run_04_stage_a_ttf_v1_gpu0.bat"
echo === 04_vq_font finished errorlevel=%errorlevel%   %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === 05_qt_font Stage A TTF v1   %DATE% %TIME% >> "%CHAIN_LOG%"
call "%REPO%\papers\05_qt_font\scripts\run_05_stage_a_ttf_v1_gpu0.bat"
echo === 05_qt_font finished errorlevel=%errorlevel%   %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === 06_calliffusion Stage A TTF v1   %DATE% %TIME% >> "%CHAIN_LOG%"
call "%REPO%\papers\06_calliffusion\scripts\run_06_stage_a_ttf_v1_gpu0.bat"
echo === 06_calliffusion finished errorlevel=%errorlevel%   %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === 08_dp_font Stage A TTF v1   %DATE% %TIME% >> "%CHAIN_LOG%"
call "%REPO%\papers\08_dp_font\scripts\run_08_stage_a_ttf_v1_gpu0.bat"
echo === 08_dp_font finished errorlevel=%errorlevel%   %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === overnight 03-08 stage A chain done %DATE% %TIME% === >> "%CHAIN_LOG%"
endlocal
