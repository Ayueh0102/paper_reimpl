@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
REM Weekend chain cuda:1 phase 2 — wait for 08 extra-long to finish, then trigger extra-extra-long.
REM Independent of cuda:0 chain. Designed to keep cuda:1 busy from Sat evening through Monday morning.

set LOG_DIR=D:\Char\ayueh\paper_reimpl\logs
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set SUP=%LOG_DIR%\weekend_chain_cuda1_phase2_%DT%.log
echo === cuda:1 phase 2 supervisor started %DATE% %TIME% === > "%SUP%"

REM Wait for 08 SB extra-long to finish.
echo --- waiting for 08 extra-long --- >> "%SUP%"
:wait_extra_long
timeout /t 300 /nobreak >nul
powershell -NoProfile -Command "$f = Get-ChildItem '%LOG_DIR%\08_stage_b_extra_long_gpu1_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f -and (Get-Content $f.FullName -Raw 2>$null) -match 'saved checkpoint -> D:\\\\Char\\\\ayueh\\\\paper_reimpl\\\\repo\\\\papers\\\\08_dp_font\\\\outputs\\\\stage_b_extra_long|Traceback') { exit 0 } else { exit 1 }"
if errorlevel 1 goto wait_extra_long
echo %DATE% %TIME% 08 extra-long ended >> "%SUP%"

REM Trigger extra-extra-long.
echo --- triggering 08 SB extra-extra-long 200k --- >> "%SUP%"
schtasks /Run /TN paper_08_stage_b_extra_extra_long_gpu1 >> "%SUP%" 2>&1
:wait_extra_extra_long
timeout /t 600 /nobreak >nul
powershell -NoProfile -Command "$f = Get-ChildItem '%LOG_DIR%\08_stage_b_extra_extra_long_gpu1_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f -and (Get-Content $f.FullName -Raw 2>$null) -match 'saved checkpoint -> D:\\\\Char\\\\ayueh\\\\paper_reimpl\\\\repo\\\\papers\\\\08_dp_font\\\\outputs\\\\stage_b_extra_extra_long|Traceback') { exit 0 } else { exit 1 }"
if errorlevel 1 goto wait_extra_extra_long
echo %DATE% %TIME% 08 extra-extra-long ended >> "%SUP%"

echo === cuda:1 phase 2 done %DATE% %TIME% === >> "%SUP%"
endlocal
