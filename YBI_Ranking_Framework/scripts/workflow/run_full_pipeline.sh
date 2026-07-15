#!/usr/bin/env bash
set -euo pipefail

# Keep one workflow implementation so that artifact paths and resume checks cannot
# drift between the cross-platform and shell entry points.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../internal/_common.sh"

exec "$CVRANK_PYTHON" "$SCRIPT_DIR/run_full_pipeline.py" "$@"
