@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
REM Overnight Stage B chain 2026-05-14.
REM
REM Waits for current cuda:0 (08 v3) + cuda:1 (04 v3 chain) jobs to finish,
REM then runs Stage B baselines on top 4 papers + 08 PINN ablation.
REM
REM cuda:0 sequential: 08 SB -> 02 SB -> 08 SB pinn0 ablation
REM cuda:1 sequential: 01 SB -> 05 SB
REM
REM Polls log markers every 60s. errorlevel checked but ignored — chain
REM continues even if a paper crashes (manifest collate mismatches expected
REM in Phase 1 Stage B placeholders).

set LOG_DIR=D:\Char\ayueh\paper_reimpl\logs
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set SUP_LOG=%LOG_DIR%\overnight_stage_b_chain_%DT%.log

echo === stage_b chain started %DATE% %TIME% === > "%SUP_LOG%"

REM ============================================
REM Phase 1: wait for 08 v3 (cuda:0) + 04 v3 (cuda:1) finishes
REM ============================================
echo waiting for 08 v3 + 04 v3 done... >> "%SUP_LOG%"

:wait_08_v3
timeout /t 60 /nobreak >nul
powershell -NoProfile -Command "$logs = Get-ChildItem '%LOG_DIR%\08_stage_a_ttf_v3_gpu0_*.log' -ErrorAction SilentlyContinue; if ($logs) { $hit = Select-String -Path $logs.FullName -Pattern 'final_step=200000' -ErrorAction SilentlyContinue; if ($hit) { exit 0 } else { exit 1 } } else { exit 1 }"
if errorlevel 1 goto wait_08_v3
echo %DATE% %TIME% 08 v3 done >> "%SUP_LOG%"

:wait_04_v3
timeout /t 60 /nobreak >nul
powershell -NoProfile -Command "$logs = Get-ChildItem '%LOG_DIR%\04_chain_v3_*.log' -ErrorAction SilentlyContinue; if ($logs) { $hit = Select-String -Path $logs.FullName -Pattern 'chain done' -ErrorAction SilentlyContinue; if ($hit) { exit 0 } else { exit 1 } } else { exit 1 }"
if errorlevel 1 goto wait_04_v3
echo %DATE% %TIME% 04 v3 chain done >> "%SUP_LOG%"

REM ============================================
REM Phase 2: launch first slot of each GPU in parallel
REM ============================================
echo %DATE% %TIME% triggering 08 SB on cuda:0 + 01 SB on cuda:1 >> "%SUP_LOG%"
schtasks /Run /TN paper_08_stage_b_gpu0 >> "%SUP_LOG%" 2>&1
schtasks /Run /TN paper_01_stage_b_gpu1 >> "%SUP_LOG%" 2>&1

REM Wait both done
:wait_08_sb
timeout /t 60 /nobreak >nul
powershell -NoProfile -Command "$logs = Get-ChildItem '%LOG_DIR%\08_stage_b_gpu0_*.log' -ErrorAction SilentlyContinue; if ($logs) { $hit = Select-String -Path $logs.FullName -Pattern 'final_step=5000|saved checkpoint|Traceback' -ErrorAction SilentlyContinue; if ($hit) { exit 0 } else { exit 1 } } else { exit 1 }"
if errorlevel 1 goto wait_08_sb
echo %DATE% %TIME% 08 SB done >> "%SUP_LOG%"
schtasks /Run /TN paper_02_stage_b_gpu0 >> "%SUP_LOG%" 2>&1

:wait_01_sb
timeout /t 60 /nobreak >nul
powershell -NoProfile -Command "$logs = Get-ChildItem '%LOG_DIR%\01_stage_b_gpu1_*.log' -ErrorAction SilentlyContinue; if ($logs) { $hit = Select-String -Path $logs.FullName -Pattern 'final_step=5000|saved checkpoint|Traceback' -ErrorAction SilentlyContinue; if ($hit) { exit 0 } else { exit 1 } } else { exit 1 }"
if errorlevel 1 goto wait_01_sb
echo %DATE% %TIME% 01 SB done >> "%SUP_LOG%"
schtasks /Run /TN paper_05_stage_b_gpu1 >> "%SUP_LOG%" 2>&1

REM ============================================
REM Phase 3: wait second slots
REM ============================================
:wait_02_sb
timeout /t 60 /nobreak >nul
powershell -NoProfile -Command "$logs = Get-ChildItem '%LOG_DIR%\02_stage_b_gpu0_*.log' -ErrorAction SilentlyContinue; if ($logs) { $hit = Select-String -Path $logs.FullName -Pattern 'final_step=5000|saved checkpoint|Traceback' -ErrorAction SilentlyContinue; if ($hit) { exit 0 } else { exit 1 } } else { exit 1 }"
if errorlevel 1 goto wait_02_sb
echo %DATE% %TIME% 02 SB done >> "%SUP_LOG%"
schtasks /Run /TN paper_08_stage_b_pinn0_gpu0 >> "%SUP_LOG%" 2>&1

:wait_05_sb
timeout /t 60 /nobreak >nul
powershell -NoProfile -Command "$logs = Get-ChildItem '%LOG_DIR%\05_stage_b_gpu1_*.log' -ErrorAction SilentlyContinue; if ($logs) { $hit = Select-String -Path $logs.FullName -Pattern 'training done|saved checkpoint|Traceback' -ErrorAction SilentlyContinue; if ($hit) { exit 0 } else { exit 1 } } else { exit 1 }"
if errorlevel 1 goto wait_05_sb
echo %DATE% %TIME% 05 SB done >> "%SUP_LOG%"

REM Phase 4: 08 pinn0 ablation
:wait_08_sb_pinn0
timeout /t 60 /nobreak >nul
powershell -NoProfile -Command "$logs = Get-ChildItem '%LOG_DIR%\08_stage_b_pinn0_gpu0_*.log' -ErrorAction SilentlyContinue; if ($logs) { $hit = Select-String -Path $logs.FullName -Pattern 'final_step=5000|saved checkpoint|Traceback' -ErrorAction SilentlyContinue; if ($hit) { exit 0 } else { exit 1 } } else { exit 1 }"
if errorlevel 1 goto wait_08_sb_pinn0
echo %DATE% %TIME% 08 SB PINN=0 done >> "%SUP_LOG%"

echo === stage_b chain done %DATE% %TIME% === >> "%SUP_LOG%"
endlocal
