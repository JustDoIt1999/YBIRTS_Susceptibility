#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../internal/_common.sh"

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/folds/run_summarize.sh [route|manifest_path]

Behavior:
  No argument   Summarize all rank-aware manifests in order:
                dbscan, no_cluster, basin
  route name    Summarize only that route. Supported names:
                dbscan, no_cluster, basin
  manifest path Summarize only the manifest you pass in.
EOF
}

run_one() {
  local manifest_path="$1"
  shift || true
  run_cvrank summarize --manifest "$manifest_path" "$@"
}

run_route() {
  local route="$1"
  shift || true
  echo "Summarizing manifest for route=$route"
  run_one "$(fold_manifest_for_route "$route")" "$@"
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
    manifest_path="$1"
    shift
    run_one "$manifest_path" "$@"
    ;;
esac
