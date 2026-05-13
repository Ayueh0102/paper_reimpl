@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
REM Overnight supervisor 2026-05-13.
REM Chains:
REM   1. wait for 08 v2 200k done -> trigger paper_05_v2_30k_gpu1 schtask
REM   2. wait for 02 v5 100k done -> trigger paper_02_sample_v5_gpu0 schtask
REM   3. wait for 05 v2 30k done  -> just log; 05 sample script not ready
REM
REM Polls log markers every 60s. Markers chosen from observed train.py output:
REM   08: "done. final_step=200000"
REM   02 v5: "saved checkpoint"
REM   05 v2: "training done; total_steps="

set LOG_DIR=D:\Char\ayueh\paper_reimpl\logs
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set SUP_LOG=%LOG_DIR%\overnight_supervisor_%DT%.log

echo === supervisor started %DATE% %TIME% === > "%SUP_LOG%"

REM -----------------------------------------------------------------
REM 1. Wait for 08 v2 done -> trigger 05 v2 30k
REM -----------------------------------------------------------------
echo waiting for 08 v2 done... >> "%SUP_LOG%"
:wait_08
timeout /t 60 /nobreak >nul
powershell -NoProfile -Command "$logs = Get-ChildItem '%LOG_DIR%\08_stage_a_ttf_v2_*.log' -ErrorAction SilentlyContinue; if ($logs) { $hit = Select-String -Path $logs.FullName -Pattern 'final_step=200000|training done' -ErrorAction SilentlyContinue; if ($hit) { exit 0 } else { exit 1 } } else { exit 1 }"
if errorlevel 1 goto wait_08
echo %DATE% %TIME% 08 v2 done detected >> "%SUP_LOG%"
schtasks /Run /TN paper_05_v2_30k_gpu1 >> "%SUP_LOG%" 2>&1
echo %DATE% %TIME% triggered paper_05_v2_30k_gpu1 >> "%SUP_LOG%"

REM -----------------------------------------------------------------
REM 2. Wait for 02 v5 done -> trigger 02 v5 sample on cuda:0
REM -----------------------------------------------------------------
echo waiting for 02 v5 done... >> "%SUP_LOG%"
:wait_02v5
timeout /t 60 /nobreak >nul
powershell -NoProfile -Command "$logs = Get-ChildItem '%LOG_DIR%\02_stage_a_ttf_v5_gpu0_*.log' -ErrorAction SilentlyContinue; if ($logs) { $hit = Select-String -Path $logs.FullName -Pattern 'done; final_step=100000|saved checkpoint' -ErrorAction SilentlyContinue; if ($hit) { exit 0 } else { exit 1 } } else { exit 1 }"
if errorlevel 1 goto wait_02v5
echo %DATE% %TIME% 02 v5 done detected >> "%SUP_LOG%"
schtasks /Run /TN paper_02_sample_v5_gpu0 >> "%SUP_LOG%" 2>&1
echo %DATE% %TIME% triggered paper_02_sample_v5_gpu0 >> "%SUP_LOG%"

REM -----------------------------------------------------------------
REM 3. Wait for 05 v2 done -> just log (sample script TBD)
REM -----------------------------------------------------------------
echo waiting for 05 v2 30k done... >> "%SUP_LOG%"
:wait_05v2
timeout /t 60 /nobreak >nul
powershell -NoProfile -Command "$logs = Get-ChildItem '%LOG_DIR%\05_stage_a_ttf_v2_*.log' -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt (Get-Date).AddHours(-24) }; if ($logs) { $hit = Select-String -Path $logs.FullName -Pattern 'training done|saved checkpoint' -ErrorAction SilentlyContinue; if ($hit) { exit 0 } else { exit 1 } } else { exit 1 }"
if errorlevel 1 goto wait_05v2
echo %DATE% %TIME% 05 v2 30k done detected >> "%SUP_LOG%"

echo === supervisor finished %DATE% %TIME% === >> "%SUP_LOG%"
endlocal
