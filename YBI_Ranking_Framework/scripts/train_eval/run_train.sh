#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../internal/_common.sh"

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_eval/run_train.sh [route|config_path]

Behavior:
  No argument   Run train for all rank-aware routes in order:
                dbscan, no_cluster, basin
  route name    Run only that route. Supported names:
                dbscan, no_cluster, basin
  config path   Run only the config you pass in.
EOF
}

run_one() {
  local config_path="$1"
  shift || true
  run_cvrank train --config "$config_path" "$@"
}

run_route() {
  local route="$1"
  shift || true
  echo "Running train for route=$route"
  run_one "$(train_validate_config_for_route "$route")" "$@"
}

if [[ $# -eq 0 ]]; then
  for route in dbscan no_cluster basin; do
    run_route "$route"
  done
  exit 0
fi

case "$1" in
  -h|--help)
    print_usage
    exit 0
    ;;
  dbscan|no_cluster|basin)
    route="$1"
    shift
    run_route "$route" "$@"
    ;;
  *)
    config_path="$1"
    shift
    run_one "$config_path" "$@"
    ;;
esac
