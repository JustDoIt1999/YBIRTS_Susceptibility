#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../internal/_common.sh"

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/lambda/run_sweep_lambda.sh [config_path] [extra_args...]

Behavior:
  No argument   Run with the default sweep config:
                configs/lambda/lambda_sweep.yaml
  config path   Run with the config you pass in.
  extra args    Passed through to `python -m cvrank sweep-lambda`
EOF
}

case "${1:-}" in
  -h|--help)
    print_usage
    exit 0
    ;;
esac

CONFIG_PATH="$CVRANK_ROOT/configs/lambda/lambda_sweep.yaml"
if [[ $# -gt 0 && "${1:-}" != -* ]]; then
  CONFIG_PATH="$1"
  shift
fi
run_cvrank sweep-lambda --config "$CONFIG_PATH" "$@"
