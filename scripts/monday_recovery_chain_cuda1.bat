@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
REM Monday recovery chain (cuda:1 only). Sequentially runs:
REM   1. 01 FontDiffuser SB-long 100k
REM   2. 04 VQ-Font     SB-long 70k
REM   3. 02 HFH-Font    SB-long 80k
REM Detection: "saved checkpoint -> D:\Char\..." or "Traceback".
REM Regex fix: single \\ in PowerShell (matches one literal \).

set LOG_DIR=D:\Char\ayueh\paper_reimpl\logs
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set SUP=%LOG_DIR%\monday_recovery_chain_%DT%.log
echo === recovery chain started %DATE% %TIME% === > "%SUP%"

echo --- phase 1: 01 SB-long 100k --- >> "%SUP%"
schtasks /Run /TN paper_01_stage_b_long_gpu1 >> "%SUP%" 2>&1
:wait_01
timeout /t 180 /nobreak >nul
powershell -NoProfile -Command "$f = Get-ChildItem '%LOG_DIR%\01_stage_b_long_gpu1_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f -and (Get-Content $f.FullName -Raw 2>$null) -match 'saved checkpoint -> D:\\Char\\ayueh\\paper_reimpl\\repo\\papers\\01_fontdiffuser\\outputs\\stage_b_long|Traceback') { exit 0 } else { exit 1 }"
if errorlevel 1 goto wait_01
echo %DATE% %TIME% 01 SB-long done >> "%SUP%"

echo --- phase 2: 04 SB-long 70k --- >> "%SUP%"
schtasks /Run /TN paper_04_stage_b_long_gpu1 >> "%SUP%" 2>&1
:wait_04
timeout /t 180 /nobreak >nul
powershell -NoProfile -Command "$f = Get-ChildItem '%LOG_DIR%\04_stage_b_long_gpu1_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f -and (Get-Content $f.FullName -Raw 2>$null) -match 'saved checkpoint -> D:\\Char\\ayueh\\paper_reimpl\\repo\\papers\\04_vq_font\\outputs\\stage_b_long|Traceback') { exit 0 } else { exit 1 }"
if errorlevel 1 goto wait_04
echo %DATE% %TIME% 04 SB-long done >> "%SUP%"

echo --- phase 3: 02 SB-long 80k --- >> "%SUP%"
schtasks /Run /TN paper_02_stage_b_long_gpu1 >> "%SUP%" 2>&1
:wait_02
timeout /t 180 /nobreak >nul
powershell -NoProfile -Command "$f = Get-ChildItem '%LOG_DIR%\02_stage_b_long_gpu1_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f -and (Get-Content $f.FullName -Raw 2>$null) -match 'saved checkpoint -> D:\\Char\\ayueh\\paper_reimpl\\repo\\papers\\02_hfh_font\\outputs\\stage_b_long|Traceback') { exit 0 } else { exit 1 }"
if errorlevel 1 goto wait_02
echo %DATE% %TIME% 02 SB-long done >> "%SUP%"

echo === recovery chain done %DATE% %TIME% === >> "%SUP%"
endlocal
