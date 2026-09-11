#!/bin/zsh
# After-close pipeline. Invoked by launchd on weekdays; do not run this by
# hand unless you mean to refresh prices + features + retrain right now.
set -euo pipefail

ROOT="${STOCK_PICKER_ROOT:-/Users/michael.artz/dev/stock-picker}"
LOG_DIR="${STOCK_PICKER_LOG_DIR:-$HOME/Library/Logs/stock-picker}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/nightly.log"

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
cd "$ROOT"
export BUILD_WORKING_DIRECTORY="$ROOT"

{
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
  bazelisk run //python/stock_picker/training:nightly
  echo "===== done $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
} >>"$LOG" 2>&1
