@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
REM Weekend chain 2026-05-15 -- two independent serial pipelines, one per GPU.
REM cuda:0 = catch-up papers (01 / 02 / 05 / 04) Stage B baselines (5k each).
REM cuda:1 = 08 long-form Stage B 50k + PINN=0 ablation 50k.
REM errorlevel ignored so any paper that crashes (channel mismatch / collate
REM bug) does not kill the whole chain.

set LOG_DIR=D:\Char\ayueh\paper_reimpl\logs
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set SUP=%LOG_DIR%\weekend_chain_%DT%.log
echo === weekend chain started %DATE% %TIME% === > "%SUP%"

REM ==========================================
REM cuda:1 pipeline (08 long-form)
REM ==========================================
echo --- cuda:1 phase 1: 08 SB long 50k --- >> "%SUP%"
schtasks /Run /TN paper_08_stage_b_long_gpu1 >> "%SUP%" 2>&1
:wait_08_long
timeout /t 120 /nobreak >nul
powershell -NoProfile -Command "$f = Get-ChildItem '%LOG_DIR%\08_stage_b_long_gpu1_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f -and (Get-Content $f.FullName -Raw 2>$null) -match 'final_step=50000|training done|saved checkpoint -> D:\\\\Char\\\\ayueh\\\\paper_reimpl\\\\repo\\\\papers\\\\08_dp_font\\\\outputs\\\\stage_b_long|Traceback') { exit 0 } else { exit 1 }"
if errorlevel 1 goto wait_08_long
echo %DATE% %TIME% 08 SB long done >> "%SUP%"

echo --- cuda:1 phase 2: 08 PINN=0 ablation 50k --- >> "%SUP%"
schtasks /Run /TN paper_08_stage_b_long_pinn0_gpu1 >> "%SUP%" 2>&1
:wait_08_pinn0_long
timeout /t 120 /nobreak >nul
powershell -NoProfile -Command "$f = Get-ChildItem '%LOG_DIR%\08_stage_b_long_pinn0_gpu1_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f -and (Get-Content $f.FullName -Raw 2>$null) -match 'final_step=50000|saved checkpoint -> D:\\\\Char\\\\ayueh\\\\paper_reimpl\\\\repo\\\\papers\\\\08_dp_font\\\\outputs\\\\stage_b_long_pinn0|Traceback') { exit 0 } else { exit 1 }"
if errorlevel 1 goto wait_08_pinn0_long
echo %DATE% %TIME% 08 SB long pinn0 done >> "%SUP%"

REM ==========================================
REM cuda:0 pipeline (other paper baselines)
REM ==========================================
echo --- cuda:0 phase 1: 01 SB 5k --- >> "%SUP%"
schtasks /Run /TN paper_01_stage_b_gpu0 >> "%SUP%" 2>&1
:wait_01_sb
timeout /t 120 /nobreak >nul
powershell -NoProfile -Command "$f = Get-ChildItem '%LOG_DIR%\01_stage_b_gpu0_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f -and (Get-Content $f.FullName -Raw 2>$null) -match 'final_step=5000|saved checkpoint|Traceback') { exit 0 } else { exit 1 }"
if errorlevel 1 goto wait_01_sb
echo %DATE% %TIME% 01 SB done >> "%SUP%"

echo --- cuda:0 phase 2: 02 SB 5k --- >> "%SUP%"
schtasks /Run /TN paper_02_stage_b_gpu0 >> "%SUP%" 2>&1
:wait_02_sb
timeout /t 120 /nobreak >nul
powershell -NoProfile -Command "$f = Get-ChildItem '%LOG_DIR%\02_stage_b_gpu0_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f -and (Get-Content $f.FullName -Raw 2>$null) -match 'final_step=5000|saved checkpoint|Traceback') { exit 0 } else { exit 1 }"
if errorlevel 1 goto wait_02_sb
echo %DATE% %TIME% 02 SB done >> "%SUP%"

echo --- cuda:0 phase 3: 05 SB 5k --- >> "%SUP%"
schtasks /Run /TN paper_05_stage_b_gpu0 >> "%SUP%" 2>&1
:wait_05_sb
timeout /t 120 /nobreak >nul
powershell -NoProfile -Command "$f = Get-ChildItem '%LOG_DIR%\05_stage_b_gpu0_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f -and (Get-Content $f.FullName -Raw 2>$null) -match 'training done|saved checkpoint|Traceback') { exit 0 } else { exit 1 }"
if errorlevel 1 goto wait_05_sb
echo %DATE% %TIME% 05 SB done >> "%SUP%"

echo --- cuda:0 phase 4: 04 SB 5k --- >> "%SUP%"
schtasks /Run /TN paper_04_stage_b_gpu0 >> "%SUP%" 2>&1
:wait_04_sb
timeout /t 120 /nobreak >nul
powershell -NoProfile -Command "$f = Get-ChildItem '%LOG_DIR%\04_stage_b_gpu0_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f -and (Get-Content $f.FullName -Raw 2>$null) -match 'final_step=5000|saved checkpoint|Traceback') { exit 0 } else { exit 1 }"
if errorlevel 1 goto wait_04_sb
echo %DATE% %TIME% 04 SB done >> "%SUP%"

echo === weekend chain done %DATE% %TIME% === >> "%SUP%"
endlocal
