@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
REM Weekend long-chain phase 2 — increase Stage B step counts on both GPUs.
REM cuda:1: 08 SB extra-long 150k continuation (warm from PINN=0 50k)
REM cuda:0: wait for 05 SB to finish, then 01 SB-long 50k -> 02 SB-long 30k -> 04 SB-long 30k.
REM Detection: explicit "saved checkpoint" string OR Traceback.

set LOG_DIR=D:\Char\ayueh\paper_reimpl\logs
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set SUP=%LOG_DIR%\weekend_chain_long_%DT%.log
echo === long chain started %DATE% %TIME% === > "%SUP%"

REM cuda:1 — kick off 08 extra-long immediately
echo --- cuda:1 phase 1: 08 SB extra-long 150k --- >> "%SUP%"
schtasks /Run /TN paper_08_stage_b_extra_long_gpu1 >> "%SUP%" 2>&1

REM cuda:0 — wait for existing 05 SB job to finish first
echo --- cuda:0 phase 0: waiting for 05 SB to finish --- >> "%SUP%"
:wait_05
timeout /t 60 /nobreak >nul
powershell -NoProfile -Command "$f = Get-ChildItem '%LOG_DIR%\05_stage_b_gpu0_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f -and (Get-Content $f.FullName -Raw 2>$null) -match 'saved checkpoint|final_step|Traceback') { exit 0 } else { exit 1 }"
if errorlevel 1 goto wait_05
echo %DATE% %TIME% 05 SB ended >> "%SUP%"

REM cuda:0 phase 1: 01 SB-long 50k
echo --- cuda:0 phase 1: 01 SB-long 50k --- >> "%SUP%"
schtasks /Run /TN paper_01_stage_b_long_gpu0 >> "%SUP%" 2>&1
:wait_01_long
timeout /t 180 /nobreak >nul
powershell -NoProfile -Command "$f = Get-ChildItem '%LOG_DIR%\01_stage_b_long_gpu0_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f -and (Get-Content $f.FullName -Raw 2>$null) -match 'final_step=50000|saved checkpoint -> D:\\\\Char\\\\ayueh\\\\paper_reimpl\\\\repo\\\\papers\\\\01_fontdiffuser\\\\outputs\\\\stage_b_long|Traceback') { exit 0 } else { exit 1 }"
if errorlevel 1 goto wait_01_long
echo %DATE% %TIME% 01 SB-long done >> "%SUP%"

REM cuda:0 phase 2: 02 SB-long 30k
echo --- cuda:0 phase 2: 02 SB-long 30k --- >> "%SUP%"
schtasks /Run /TN paper_02_stage_b_long_gpu0 >> "%SUP%" 2>&1
:wait_02_long
timeout /t 180 /nobreak >nul
powershell -NoProfile -Command "$f = Get-ChildItem '%LOG_DIR%\02_stage_b_long_gpu0_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f -and (Get-Content $f.FullName -Raw 2>$null) -match 'final_step=30000|saved checkpoint -> D:\\\\Char\\\\ayueh\\\\paper_reimpl\\\\repo\\\\papers\\\\02_hfh_font\\\\outputs\\\\stage_b_long|Traceback') { exit 0 } else { exit 1 }"
if errorlevel 1 goto wait_02_long
echo %DATE% %TIME% 02 SB-long done >> "%SUP%"

REM cuda:0 phase 3: 04 SB-long 30k
echo --- cuda:0 phase 3: 04 SB-long 30k --- >> "%SUP%"
schtasks /Run /TN paper_04_stage_b_long_gpu0 >> "%SUP%" 2>&1
:wait_04_long
timeout /t 180 /nobreak >nul
powershell -NoProfile -Command "$f = Get-ChildItem '%LOG_DIR%\04_stage_b_long_gpu0_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f -and (Get-Content $f.FullName -Raw 2>$null) -match 'final_step=30000|saved checkpoint -> D:\\\\Char\\\\ayueh\\\\paper_reimpl\\\\repo\\\\papers\\\\04_vq_font\\\\outputs\\\\stage_b_long|Traceback') { exit 0 } else { exit 1 }"
if errorlevel 1 goto wait_04_long
echo %DATE% %TIME% 04 SB-long done >> "%SUP%"

echo === long chain done %DATE% %TIME% === >> "%SUP%"
endlocal
