#!/usr/bin/env bash
# Fetch 二南堂 sample grids from the lab server, then compose the
# side-by-side comparison PNG and open it.
#
# Run on Monday morning (Mac side):
#   LAB_SSH_HOST=... LAB_SSH_USER=... LAB_REMOTE_REPO_ROOT=... \
#     bash scripts/fetch_weekend_results.sh
#
# Continues on failure: per-paper scp errors print a WARN line so we
# still get a partial composite if some papers haven't finished training.

set -u

: "${LAB_SSH_HOST:?Set LAB_SSH_HOST to the lab server hostname or IP.}"
: "${LAB_SSH_USER:?Set LAB_SSH_USER to the lab server username.}"
: "${LAB_REMOTE_REPO_ROOT:?Set LAB_REMOTE_REPO_ROOT to the paper_reimpl repo path on the lab server.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${LAB_SSH_USER}@${LAB_SSH_HOST}"
LAB_SSH_PORT="${LAB_SSH_PORT:-22}"
SSHPASS_BIN="${SSHPASS_BIN:-sshpass}"
SCP_OPTS=(
  -P "$LAB_SSH_PORT"
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
)

scp_from_lab() {
  local src="$1"
  local dst="$2"
  local ssh_password="${LAB_SSH_PASS:-${LAB_SSH_PASSWORD:-}}"
  if [[ -n "$ssh_password" ]]; then
    if ! command -v "$SSHPASS_BIN" >/dev/null 2>&1; then
      echo "ERROR: sshpass not found. Install with: brew install hudochenkov/sshpass/sshpass"
      exit 1
    fi
    "$SSHPASS_BIN" -p "$ssh_password" scp "${SCP_OPTS[@]}" "$SERVER:$src" "$dst"
  else
    scp "${SCP_OPTS[@]}" "$SERVER:$src" "$dst"
  fi
}

# server_path  ->  local_path  (separated by '|')
TRANSFERS=(
  "$LAB_REMOTE_REPO_ROOT/papers/01_fontdiffuser/outputs/stage_b_long/sample_grid_ernantang.png|/tmp/01_sb_ernantang.png"
  "$LAB_REMOTE_REPO_ROOT/papers/02_hfh_font/outputs/stage_b_long/sample_grid_ernantang.png|/tmp/02_sb_ernantang.png"
  "$LAB_REMOTE_REPO_ROOT/papers/04_vq_font/outputs/stage_b_long/sample_grid_ernantang.png|/tmp/04_sb_ernantang.png"
  "$LAB_REMOTE_REPO_ROOT/papers/05_qt_font/outputs/stage_b/sample_grid_ernantang.png|/tmp/05_sb_ernantang.png"
  "$LAB_REMOTE_REPO_ROOT/papers/08_dp_font/outputs/stage_b_extra_long/sample_grid_ernantang.png|/tmp/08_sb_extra_long_sample.png"
  "$LAB_REMOTE_REPO_ROOT/papers/08_dp_font/outputs/stage_b_extra_extra_long/sample_grid_ernantang.png|/tmp/08_sb_extra_extra_long_sample.png"
)

echo "[fetch] pulling sample grids from $SERVER ..."
for entry in "${TRANSFERS[@]}"; do
  src="${entry%%|*}"
  dst="${entry##*|}"
  echo "  scp $src -> $dst"
  scp_from_lab "$src" "$dst" \
    || echo "WARN: scp failed for $src (file may not exist yet)"
done

# Also try the 50k-PINN=0 baseline grid if a previous run left it.
# Filename on Mac side matches the panel definition in compose_paper_comparison.py.
BASELINE_SRC="$LAB_REMOTE_REPO_ROOT/papers/08_dp_font/outputs/stage_b_long_pinn0/sample_grid_ernantang.png"
BASELINE_DST="/tmp/08_sb_50k_pinn0_sample.png"
echo "  scp $BASELINE_SRC -> $BASELINE_DST  (optional baseline)"
scp_from_lab "$BASELINE_SRC" "$BASELINE_DST" \
  || echo "WARN: scp failed for $BASELINE_SRC (optional)"

echo "[fetch] composing comparison PNG ..."
python3 "$SCRIPT_DIR/compose_paper_comparison.py" \
  --samples-dir /tmp \
  --output /tmp/paper_comparison.png \
  --row-height 100

echo "[fetch] opening in Preview ..."
open /tmp/paper_comparison.png
