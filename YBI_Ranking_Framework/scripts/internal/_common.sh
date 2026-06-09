#!/usr/bin/env bash
set -euo pipefail

CVRANK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  CVRANK_PYTHON="$PYTHON_BIN"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  CVRANK_PYTHON="${CONDA_PREFIX}/bin/python"
else
  CVRANK_PYTHON="$(command -v python)"
fi

run_cvrank() {
  PYTHONPATH="$CVRANK_ROOT/src" "$CVRANK_PYTHON" -m cvrank "$@"
}

route_config_for_route() {
  case "${1:-}" in
    dbscan)
      echo "$CVRANK_ROOT/configs/routes/dbscan.yaml"
      ;;
    no_cluster)
      echo "$CVRANK_ROOT/configs/routes/no_cluster.yaml"
      ;;
    basin)
      echo "$CVRANK_ROOT/configs/routes/basin.yaml"
      ;;
    *)
      echo "Unsupported route: ${1:-}" >&2
      return 1
      ;;
  esac
}

train_validate_config_for_route() {
  route_config_for_route "$@"
}

fold_manifest_for_route() {
  case "${1:-}" in
    dbscan)
      echo "$CVRANK_ROOT/data/output/folds/dbscan_5fold_manifest.csv"
      ;;
    no_cluster)
      echo "$CVRANK_ROOT/data/output/folds/no_cluster_5fold_manifest.csv"
      ;;
    basin)
      echo "$CVRANK_ROOT/data/output/folds/basin_5fold_manifest.csv"
      ;;
    *)
      echo "Unsupported route: ${1:-}" >&2
      return 1
      ;;
  esac
}
