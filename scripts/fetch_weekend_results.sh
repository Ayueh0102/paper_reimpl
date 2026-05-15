#!/usr/bin/env bash
# Fetch 二南堂 sample grids from the lab server, then compose the
# side-by-side comparison PNG and open it.
#
# Run on Monday morning (Mac side):
#   bash /Users/Ayueh/Char/paper_reimpl/scripts/fetch_weekend_results.sh
#
# Continues on failure: per-paper scp errors print a WARN line so we
# still get a partial composite if some papers haven't finished training.

set -u

SERVER="ptri@10.102.10.212"
PASS='ptri78322626'
SSHPASS_BIN="${SSHPASS_BIN:-sshpass}"
SCP_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

# server_path  ->  local_path  (separated by '|')
TRANSFERS=(
  "D:/Char/ayueh/paper_reimpl/repo/papers/01_fontdiffuser/outputs/stage_b_long/sample_grid_ernantang.png|/tmp/01_sb_ernantang.png"
  "D:/Char/ayueh/paper_reimpl/repo/papers/02_hfh_font/outputs/stage_b_long/sample_grid_ernantang.png|/tmp/02_sb_ernantang.png"
  "D:/Char/ayueh/paper_reimpl/repo/papers/04_vq_font/outputs/stage_b_long/sample_grid_ernantang.png|/tmp/04_sb_ernantang.png"
  "D:/Char/ayueh/paper_reimpl/repo/papers/05_qt_font/outputs/stage_b/sample_grid_ernantang.png|/tmp/05_sb_ernantang.png"
  "D:/Char/ayueh/paper_reimpl/repo/papers/08_dp_font/outputs/stage_b_extra_long/sample_grid_ernantang.png|/tmp/08_sb_extra_long_sample.png"
  "D:/Char/ayueh/paper_reimpl/repo/papers/08_dp_font/outputs/stage_b_extra_extra_long/sample_grid_ernantang.png|/tmp/08_sb_extra_extra_long_sample.png"
)

if ! command -v "$SSHPASS_BIN" >/dev/null 2>&1; then
  echo "ERROR: sshpass not found. Install with: brew install hudochenkov/sshpass/sshpass"
  exit 1
fi

echo "[fetch] pulling sample grids from $SERVER ..."
for entry in "${TRANSFERS[@]}"; do
  src="${entry%%|*}"
  dst="${entry##*|}"
  echo "  scp $src -> $dst"
  "$SSHPASS_BIN" -p "$PASS" scp $SCP_OPTS "$SERVER:$src" "$dst" \
    || echo "WARN: scp failed for $src (file may not exist yet)"
done

# Also try the 50k-PINN=0 baseline grid if a previous run left it.
# Filename on Mac side matches the panel definition in compose_paper_comparison.py.
BASELINE_SRC="D:/Char/ayueh/paper_reimpl/repo/papers/08_dp_font/outputs/stage_b_long_pinn0/sample_grid_ernantang.png"
BASELINE_DST="/tmp/08_sb_50k_pinn0_sample.png"
echo "  scp $BASELINE_SRC -> $BASELINE_DST  (optional baseline)"
"$SSHPASS_BIN" -p "$PASS" scp $SCP_OPTS "$SERVER:$BASELINE_SRC" "$BASELINE_DST" \
  || echo "WARN: scp failed for $BASELINE_SRC (optional)"

echo "[fetch] composing comparison PNG ..."
python3 /Users/Ayueh/Char/paper_reimpl/scripts/compose_paper_comparison.py \
  --samples-dir /tmp \
  --output /tmp/paper_comparison.png \
  --row-height 100

echo "[fetch] opening in Preview ..."
open /tmp/paper_comparison.png
