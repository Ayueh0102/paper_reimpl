@echo off
chcp 65001 >nul
setlocal
REM 02 paper-faithful chain on cuda:1:
REM   1. VAE pretrain v2 (30000 step, ~1 hr)
REM   2. Stage A v3 (50000 step U-Net, warm-start from new VAE, ~2 hr)
REM
REM Estimated total: ~3 hr.

set REPO=D:\Char\ayueh\paper_reimpl\repo
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set CHAIN_DT=%%I
set CHAIN_LOG=D:\Char\ayueh\paper_reimpl\logs\02_chain_vae_v2_stage_a_v3_%CHAIN_DT%.log

echo === 02 chain started %DATE% %TIME% === > "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === step 1/2: VAE pretrain v2 (30000 step) === >> "%CHAIN_LOG%"
call "%REPO%\papers\02_hfh_font\scripts\run_02_vae_pretrain_v2_gpu1.bat"
echo === VAE v2 finished errorlevel=%errorlevel%   %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === step 2/2: Stage A v3 U-Net (50000 step, frozen VAE v2) === >> "%CHAIN_LOG%"
call "%REPO%\papers\02_hfh_font\scripts\run_02_stage_a_ttf_v3_gpu1.bat"
echo === Stage A v3 finished errorlevel=%errorlevel%   %DATE% %TIME% >> "%CHAIN_LOG%"

echo. >> "%CHAIN_LOG%"
echo === 02 chain done %DATE% %TIME% === >> "%CHAIN_LOG%"
endlocal
