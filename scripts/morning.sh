#!/bin/zsh
# Start Bazel before the bell. Python checks the backup switch at 8:30:05 CT,
# then collects real-time Polygon opens through 8:33 CT before REST fallbacks.
set -euo pipefail

ROOT="${STOCK_PICKER_ROOT:-/Users/michael.artz/dev/stock-picker}"
LOG_DIR="${STOCK_PICKER_LOG_DIR:-$HOME/Library/Logs/stock-picker}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/morning.log"

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
cd "$ROOT"
export BUILD_WORKING_DIRECTORY="$ROOT"

{
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
  echo "===== score $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
  bazelisk run //python/stock_picker/training:morning
  echo "===== done $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
} >>"$LOG" 2>&1
