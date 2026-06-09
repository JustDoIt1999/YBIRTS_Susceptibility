#!/usr/bin/env bash
set -euo pipefail

workflow_config_path_from_args() {
  local default_path="$1"
  shift

  local config_path="$default_path"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for --config" >&2
          return 1
        fi
        config_path="$2"
        shift
        ;;
    esac
    shift
  done
  echo "$config_path"
}

_trim_workflow_value() {
  local value="${1:-}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  if [[ "$value" == \"*\" && "$value" == *\" ]]; then
    value="${value:1:${#value}-2}"
  elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
    value="${value:1:${#value}-2}"
  fi
  echo "$value"
}

_workflow_lower() {
  printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]'
}

workflow_yaml_get_raw() {
  local config_path="$1"
  local target_key="$2"

  awk -F: -v target="$target_key" '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    {
      key = $1
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
      if (key != target) {
        next
      }
      line = $0
      sub(/^[^:]*:[[:space:]]*/, "", line)
      sub(/[[:space:]]+#.*$/, "", line)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
      print line
      exit
    }
  ' "$config_path"
}

workflow_yaml_list_keys() {
  local config_path="$1"

  awk -F: '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    {
      key = $1
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
      if (key != "") {
        print key
      }
    }
  ' "$config_path"
}

normalize_workflow_bool() {
  local key_name="$1"
  local raw_value="$2"
  local default_value="${3:-false}"
  local value
  local lower_value
  value="$(_trim_workflow_value "$raw_value")"
  lower_value="$(_workflow_lower "$value")"

  case "$lower_value" in
    "")
      echo "$default_value"
      ;;
    true|yes|y|on|1)
      echo "true"
      ;;
    false|no|n|off|0|null|~)
      echo "false"
      ;;
    *)
      echo "Invalid boolean for \`$key_name\`: $raw_value" >&2
      return 1
      ;;
  esac
}

normalize_workflow_optional_int() {
  local key_name="$1"
  local raw_value="$2"
  local value
  local lower_value
  value="$(_trim_workflow_value "$raw_value")"
  lower_value="$(_workflow_lower "$value")"

  case "$lower_value" in
    ""|null|~)
      echo ""
      return 0
      ;;
  esac

  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "Invalid integer for \`$key_name\`: $raw_value" >&2
    return 1
  fi
  if (( value <= 0 )); then
    echo "\`$key_name\` must be > 0, got $value." >&2
    return 1
  fi
  echo "$value"
}

normalize_workflow_rank_loss_mode() {
  local raw_value="$1"
  local value
  local lower_value
  value="$(_trim_workflow_value "$raw_value")"
  lower_value="$(_workflow_lower "$value")"

  case "$lower_value" in
    ""|null|~)
      echo ""
      ;;
    lambdarank|lambda|ndcg_aware_lambdarank)
      echo "ndcg_aware_lambdarank"
      ;;
    disabled|none|off)
      echo "disabled"
      ;;
    *)
      echo "Invalid \`rank_loss_mode\`: $raw_value" >&2
      return 1
      ;;
  esac
}

validate_workflow_yaml_keys() {
  local workflow_kind="$1"
  local config_path="$2"
  local invalid_keys=()
  local key=""

  while IFS= read -r key; do
    case "$workflow_kind:$key" in
      full:skip_existing|full:batch_size|full:rank_loss_mode)
        ;;
      *)
        invalid_keys+=("$key")
        ;;
    esac
  done < <(workflow_yaml_list_keys "$config_path")

  if (( ${#invalid_keys[@]} > 0 )); then
    echo "Unsupported keys for workflow '$workflow_kind' in $config_path: ${invalid_keys[*]}" >&2
    return 1
  fi
}

load_workflow_config() {
  local workflow_kind="$1"
  local config_path="$2"

  if [[ -z "$config_path" ]]; then
    return 0
  fi
  if [[ ! -f "$config_path" ]]; then
    echo "Workflow config not found: $config_path" >&2
    return 1
  fi

  validate_workflow_yaml_keys "$workflow_kind" "$config_path"

  SKIP_EXISTING="$(normalize_workflow_bool "skip_existing" "$(workflow_yaml_get_raw "$config_path" "skip_existing")" "false")"
  BATCH_SIZE="$(normalize_workflow_optional_int "batch_size" "$(workflow_yaml_get_raw "$config_path" "batch_size")")"

  if [[ "$workflow_kind" == "full" ]]; then
    RANK_LOSS_MODE="$(normalize_workflow_rank_loss_mode "$(workflow_yaml_get_raw "$config_path" "rank_loss_mode")")"
  fi
}
