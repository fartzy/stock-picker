#!/bin/zsh
# Weekday morning scoring at 8:32 CT. Writes
# $ROOT/data/buy_signals/{today}.json and latest.json -- the Trading tab
# reads that cache so 8:37 is instant. Skips if last night's pipeline is stale.
set -euo pipefail

ROOT="${STOCK_PICKER_ROOT:-/Users/michael.artz/dev/stock-picker}"
LOG_DIR="${STOCK_PICKER_LOG_DIR:-$HOME/Library/Logs/stock-picker}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/morning.log"

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
cd "$ROOT"
# Pin data_root() to this checkout even if bazel run's cwd is a sandbox.
export BUILD_WORKING_DIRECTORY="$ROOT"

{
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
  bazelisk run //python/stock_picker/training:morning
  echo "===== done $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
} >>"$LOG" 2>&1
