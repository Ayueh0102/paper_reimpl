@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
REM Canary chain (cuda:1 only). Each phase 1500 steps to verify:
REM   - manifest with non-empty ref_image_paths is reachable
REM   - writer-balanced sampler is actually wired (log writer dist if available)
REM   - 04 codebook gradient actually moves after Stage 0 fix
REM Completion detection: explicit "saved checkpoint" / "final_step" / "Traceback".

set LOG_DIR=D:\Char\ayueh\paper_reimpl\logs
set REPO=D:\Char\ayueh\paper_reimpl\repo
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I
set SUP=%LOG_DIR%\canary_chain_cuda1_%DT%.log
echo === canary chain started %DATE% %TIME% === > "%SUP%"

REM Phase 1: 08 SB canary (no refs, balanced sampler only).
echo --- phase 1: 08 SB canary 1500 --- >> "%SUP%"
set TRAIN=%REPO%\papers\08_dp_font\src\dp_font\configs\train_stage_b_canary.yaml
set MODEL=%REPO%\papers\08_dp_font\src\dp_font\configs\model.yaml
set DATA=%REPO%\papers\08_dp_font\src\dp_font\configs\data_stage_b.yaml
set INIT=%REPO%\papers\08_dp_font\outputs\stage_b_extra_long\dp_font_last.pt
set LOG=%LOG_DIR%\08_canary_%DT%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d %REPO%\papers\08_dp_font
uv run python -u -m paper_reimpl_shared.runner.entrypoint --paper dp_font --train "%TRAIN%" --model "%MODEL%" --data "%DATA%" --data-backend lab_server --device cuda:1 --init-ckpt "%INIT%" > "%LOG%" 2>&1
echo %DATE% %TIME% 08 canary done >> "%SUP%"

REM Phase 2: 01 SB canary (1 ref, balanced sampler).
echo --- phase 2: 01 SB canary 1500 --- >> "%SUP%"
set TRAIN=%REPO%\papers\01_fontdiffuser\src\fontdiffuser\configs\train_stage_b_canary.yaml
set MODEL=%REPO%\papers\01_fontdiffuser\src\fontdiffuser\configs\model.yaml
set DATA=%REPO%\papers\01_fontdiffuser\src\fontdiffuser\configs\data_stage_b.yaml
set INIT=%REPO%\papers\01_fontdiffuser\outputs\stage_b_long\fontdiffuser_last.pt
set LOG=%LOG_DIR%\01_canary_%DT%.log
cd /d %REPO%\papers\01_fontdiffuser
uv run python -u -m paper_reimpl_shared.runner.entrypoint --paper fontdiffuser --train "%TRAIN%" --model "%MODEL%" --data "%DATA%" --data-backend lab_server --device cuda:1 --init-ckpt "%INIT%" > "%LOG%" 2>&1
echo %DATE% %TIME% 01 canary done >> "%SUP%"

REM Phase 3: 02 SB canary (latent, refs).
echo --- phase 3: 02 SB canary 1500 --- >> "%SUP%"
set TRAIN=%REPO%\papers\02_hfh_font\src\hfh_font\configs\train_stage_b_canary.yaml
set MODEL=%REPO%\papers\02_hfh_font\src\hfh_font\configs\model.yaml
set DATA=%REPO%\papers\02_hfh_font\src\hfh_font\configs\data_stage_b.yaml
set INIT=%REPO%\papers\02_hfh_font\outputs\stage_b_long\hfh_font_last.pt
set LOG=%LOG_DIR%\02_canary_%DT%.log
cd /d %REPO%\papers\02_hfh_font
uv run python -u -m paper_reimpl_shared.runner.entrypoint --paper hfh_font --train "%TRAIN%" --model "%MODEL%" --data "%DATA%" --data-backend lab_server --device cuda:1 --init-ckpt "%INIT%" > "%LOG%" 2>&1
echo %DATE% %TIME% 02 canary done >> "%SUP%"

REM Phase 4: 04 Stage 0 VQGAN canary — codebook gradient must move.
echo --- phase 4: 04 Stage 0 VQGAN canary 1500 --- >> "%SUP%"
set TRAIN=%REPO%\papers\04_vq_font\src\vq_font\configs\train_stage_0_vqgan_canary.yaml
set MODEL=%REPO%\papers\04_vq_font\src\vq_font\configs\model.yaml
set DATA=%REPO%\papers\04_vq_font\src\vq_font\configs\data_stage_a_ttf.yaml
set LOG=%LOG_DIR%\04_stage0_canary_%DT%.log
cd /d %REPO%\papers\04_vq_font
uv run python -u -m paper_reimpl_shared.runner.entrypoint --paper vq_font --train "%TRAIN%" --model "%MODEL%" --data "%DATA%" --data-backend lab_server --device cuda:1 > "%LOG%" 2>&1
echo %DATE% %TIME% 04 stage0 canary done >> "%SUP%"

echo === canary chain done %DATE% %TIME% === >> "%SUP%"
endlocal
