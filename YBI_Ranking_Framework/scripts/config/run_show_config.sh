#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../internal/_common.sh"

CONFIG_PATH="${1:-$CVRANK_ROOT/configs/common.yaml}"
shift || true
run_cvrank show-config --config "$CONFIG_PATH" "$@"
