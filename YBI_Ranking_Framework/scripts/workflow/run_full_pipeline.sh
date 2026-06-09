#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../internal/_common.sh"
source "$SCRIPT_DIR/../internal/_workflow_config.sh"

SKIP_EXISTING=false
BATCH_SIZE=""
RANK_LOSS_MODE=""
WORKFLOW_CONFIG=""
OUTPUT_SUFFIX=""

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/workflow/run_full_pipeline.sh [--config PATH] [--skip-existing|--no-skip-existing] [--batch-size N|--clear-batch-size] [--rank-loss-mode MODE|--clear-rank-loss-mode] [--output-suffix NAME]

Options:
  --config PATH                        Workflow-level YAML defaults. Default:
                                       scripts/workflow/full_pipeline.yaml
  --skip-existing                      Skip steps whose key output artifacts already exist.
  --no-skip-existing                   Force re-running all steps for this run.
  --batch-size N                       Override batch size for this full rank-aware pipeline run and isolate outputs under bs_N.
  --clear-batch-size                   Ignore any workflow-level batch size override and fall back to the route config default.
  --rank-loss-mode MODE                Override the rank-aware auxiliary loss used by sweep-lambda and select-lambda --train.
                                       Supported CLI values include: lambdarank, disabled.
  --clear-rank-loss-mode               Ignore any workflow-level rank loss mode override and use the route config default.
  --output-suffix NAME                 Append a suffix to output roots, e.g. archive -> output_dbscan_archive.
  -h, --help                           Show this help message.
EOF
}

append_output_suffix_name() {
  local name="$1"
  if [[ -z "$OUTPUT_SUFFIX" ]]; then
    echo "$name"
    return 0
  fi
  echo "${name}_${OUTPUT_SUFFIX}"
}

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      print_usage
      exit 0
      ;;
  esac
done

WORKFLOW_CONFIG="$(workflow_config_path_from_args "$CVRANK_ROOT/scripts/workflow/full_pipeline.yaml" "$@")"
load_workflow_config full "$WORKFLOW_CONFIG"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --config" >&2
        print_usage >&2
        exit 1
      fi
      WORKFLOW_CONFIG="$2"
      shift
      ;;
    --skip-existing)
      SKIP_EXISTING=true
      ;;
    --no-skip-existing)
      SKIP_EXISTING=false
      ;;
    --batch-size)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --batch-size" >&2
        print_usage >&2
        exit 1
      fi
      BATCH_SIZE="$2"
      shift
      ;;
    --clear-batch-size)
      BATCH_SIZE=""
      ;;
    --rank-loss-mode)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --rank-loss-mode" >&2
        print_usage >&2
        exit 1
      fi
      RANK_LOSS_MODE="$2"
      shift
      ;;
    --clear-rank-loss-mode)
      RANK_LOSS_MODE=""
      ;;
    --output-suffix)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --output-suffix" >&2
        print_usage >&2
        exit 1
      fi
      OUTPUT_SUFFIX="$2"
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_usage >&2
      exit 1
      ;;
  esac
  shift
done

declare -a EXTRA_ARGS=()
declare -a SWEEP_EXTRA_ARGS=()
declare -a SELECT_EXTRA_ARGS=()
if [[ -n "$BATCH_SIZE" ]]; then
  EXTRA_ARGS+=(--batch-size "$BATCH_SIZE")
  SWEEP_EXTRA_ARGS+=(--batch-size "$BATCH_SIZE")
  SELECT_EXTRA_ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$OUTPUT_SUFFIX" ]]; then
  EXTRA_ARGS+=(--output-suffix "$OUTPUT_SUFFIX")
  SWEEP_EXTRA_ARGS+=(--output-suffix "$OUTPUT_SUFFIX")
  SELECT_EXTRA_ARGS+=(--output-suffix "$OUTPUT_SUFFIX")
fi
if [[ -n "$RANK_LOSS_MODE" ]]; then
  RANK_LOSS_MODE="$(normalize_workflow_rank_loss_mode "$RANK_LOSS_MODE")"
  SWEEP_EXTRA_ARGS+=(--rank-loss-mode "$RANK_LOSS_MODE")
  SELECT_EXTRA_ARGS+=(--rank-loss-mode "$RANK_LOSS_MODE")
fi

print_effective_options() {
  local effective_batch_size="<route-config-default>"
  local effective_rank_loss_mode="<route-config-default>"
  local effective_output_suffix="<none>"

  if [[ -n "$BATCH_SIZE" ]]; then
    effective_batch_size="$BATCH_SIZE"
  fi
  if [[ -n "$RANK_LOSS_MODE" ]]; then
    effective_rank_loss_mode="$RANK_LOSS_MODE"
  fi
  if [[ -n "$OUTPUT_SUFFIX" ]]; then
    effective_output_suffix="$OUTPUT_SUFFIX"
  fi

  echo "Workflow config: $WORKFLOW_CONFIG"
  echo "Effective workflow options:"
  echo "  skip_existing=$SKIP_EXISTING"
  echo "  batch_size=$effective_batch_size"
  echo "  rank_loss_mode=$effective_rank_loss_mode"
  echo "  output_suffix=$effective_output_suffix"
}

print_effective_options

run_step() {
  local title="$1"
  shift
  echo
  echo "==> ${title}"
  "$@"
}

all_paths_exist() {
  local path=""
  for path in "$@"; do
    [[ -e "$path" ]] || return 1
  done
  return 0
}

all_globs_match() {
  local pattern=""
  for pattern in "$@"; do
    compgen -G "$pattern" > /dev/null || return 1
  done
  return 0
}

run_step_if_needed() {
  local title="$1"
  shift

  local -a success_paths=()
  local -a success_globs=()
  local -a command=()
  local marker=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --paths)
        shift
        while [[ $# -gt 0 && "$1" != "--globs" && "$1" != "--" ]]; do
          success_paths+=("$1")
          shift
        done
        ;;
      --globs)
        shift
        while [[ $# -gt 0 && "$1" != "--paths" && "$1" != "--" ]]; do
          success_globs+=("$1")
          shift
        done
        ;;
      --)
        shift
        command=("$@")
        break
        ;;
      *)
        echo "run_step_if_needed internal error: unexpected argument: $1" >&2
        return 1
        ;;
    esac
  done

  if [[ ${#command[@]} -eq 0 ]]; then
    echo "run_step_if_needed internal error: missing command for step: $title" >&2
    return 1
  fi

  if [[ "$SKIP_EXISTING" == "true" ]]; then
    local paths_ok=true
    local globs_ok=true

    if [[ ${#success_paths[@]} -gt 0 ]]; then
      if ! all_paths_exist "${success_paths[@]}"; then
        paths_ok=false
      else
        marker="${success_paths[0]}"
      fi
    fi

    if [[ ${#success_globs[@]} -gt 0 ]]; then
      if ! all_globs_match "${success_globs[@]}"; then
        globs_ok=false
      elif [[ -z "$marker" ]]; then
        marker="$(compgen -G "${success_globs[0]}" | head -n 1)"
      fi
    fi

    if [[ "$paths_ok" == "true" && "$globs_ok" == "true" ]] && [[ ${#success_paths[@]} -gt 0 || ${#success_globs[@]} -gt 0 ]]; then
      echo
      echo "==> skip (${title})"
      if [[ -n "$marker" ]]; then
        echo "    already exists: $marker"
      fi
      return 0
    fi
  fi

  run_step "$title" "${command[@]}"
}

selection_output_root() {
  local root="$CVRANK_ROOT/data/output/$(append_output_suffix_name selected_lambda)"

  if [[ -n "$BATCH_SIZE" ]]; then
    root="$root/bs_${BATCH_SIZE}"
  fi
  if [[ -n "$RANK_LOSS_MODE" ]]; then
    root="$root/$RANK_LOSS_MODE"
  else
    root="$root/ndcg_aware_lambdarank"
  fi
  echo "$root"
}

selected_output_root_for_route() {
  local route="$1"
  local root="$CVRANK_ROOT/data/output/$(append_output_suffix_name "output_${route}_selected")"

  if [[ -n "$BATCH_SIZE" ]]; then
    root="$root/bs_${BATCH_SIZE}"
  fi
  if [[ -n "$RANK_LOSS_MODE" ]]; then
    root="$root/$RANK_LOSS_MODE"
  else
    root="$root/ndcg_aware_lambdarank"
  fi
  echo "$root"
}

run_make_folds() {
  local route="$1"
  local config_path="$2"
  local manifest_path="$CVRANK_ROOT/data/output/folds/${route}_5fold_manifest.csv"
  run_step_if_needed \
    "make-folds (${route})" \
    --paths "$manifest_path" \
    -- \
    run_cvrank make-folds --config "$config_path"
}

run_make_folds "dbscan" "$CVRANK_ROOT/configs/routes/dbscan.yaml"
run_make_folds "no_cluster" "$CVRANK_ROOT/configs/routes/no_cluster.yaml"
run_make_folds "basin" "$CVRANK_ROOT/configs/routes/basin.yaml"

declare -a sweep_cmd=(run_cvrank sweep-lambda --config "$CVRANK_ROOT/configs/lambda/lambda_sweep.yaml")
if [[ ${#SWEEP_EXTRA_ARGS[@]} -gt 0 ]]; then
  sweep_cmd+=("${SWEEP_EXTRA_ARGS[@]}")
fi

run_step_if_needed \
  "sweep-lambda" \
  --globs \
    "$CVRANK_ROOT/data/output/sweeps/$(append_output_suffix_name lambda_rank)${BATCH_SIZE:+/bs_${BATCH_SIZE}}/${RANK_LOSS_MODE:-ndcg_aware_lambdarank}/lambda_sweep_runs*.csv" \
    "$CVRANK_ROOT/data/output/sweeps/$(append_output_suffix_name lambda_rank)${BATCH_SIZE:+/bs_${BATCH_SIZE}}/${RANK_LOSS_MODE:-ndcg_aware_lambdarank}/lambda_sweep_summary*.csv" \
    "$CVRANK_ROOT/data/output/sweeps/$(append_output_suffix_name lambda_rank)${BATCH_SIZE:+/bs_${BATCH_SIZE}}/${RANK_LOSS_MODE:-ndcg_aware_lambdarank}/lambda_sweep_manifest*.json" \
  -- \
  "${sweep_cmd[@]}"

selection_output_root_path="$(selection_output_root)"
dbscan_selected_output_root="$(selected_output_root_for_route dbscan)"
no_cluster_selected_output_root="$(selected_output_root_for_route no_cluster)"
basin_selected_output_root="$(selected_output_root_for_route basin)"

declare -a select_cmd=(run_cvrank select-lambda --config "$CVRANK_ROOT/configs/lambda/lambda_select.yaml" --train)
if [[ ${#SELECT_EXTRA_ARGS[@]} -gt 0 ]]; then
  select_cmd+=("${SELECT_EXTRA_ARGS[@]}")
fi

run_step_if_needed \
  "select-lambda (train)" \
  --paths \
    "$selection_output_root_path/selection_manifest.json" \
    "$dbscan_selected_output_root/selection/dbscan_best_train.yaml" \
    "$no_cluster_selected_output_root/selection/no_cluster_best_train.yaml" \
    "$basin_selected_output_root/selection/basin_best_train.yaml" \
  --globs \
    "$dbscan_selected_output_root/fold_runs_manifest*.json" \
    "$no_cluster_selected_output_root/fold_runs_manifest*.json" \
    "$basin_selected_output_root/fold_runs_manifest*.json" \
  -- \
  "${select_cmd[@]}"
