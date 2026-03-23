#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_diagnose_experts_proxy.sh [QUANT_DIR] [PROXY_LOG] [JSON_OUT]
#
# Examples:
#   bash scripts/run_diagnose_experts_proxy.sh
#   bash scripts/run_diagnose_experts_proxy.sh /path/to/save_path
#   bash scripts/run_diagnose_experts_proxy.sh /path/to/save_path experts_proxy.log

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_SCRIPT="${ROOT_DIR}/scripts/diagnose_experts_proxy.py"

cd "${ROOT_DIR}"

EXPERTS_HESSIAN_DIR="${EXPERTS_HESSIAN_DIR:-${ROOT_DIR}/hessians_deepseek_moe_16b_base_experts}"
DEVICE="${DEVICE:-auto}"
SAMPLE_SIZE="${SAMPLE_SIZE:-300}"

QUANT_DIR="${1:-}"
PROXY_LOG="${2:-}"
JSON_OUT="${3:-}"

CMD=(python "${PY_SCRIPT}" \
  --experts-hessian-dir "${EXPERTS_HESSIAN_DIR}" \
  --device "${DEVICE}" \
  --sample-size "${SAMPLE_SIZE}")

if [[ -n "${QUANT_DIR}" ]]; then
  CMD+=(--quant-dir "${QUANT_DIR}")
fi

if [[ -n "${PROXY_LOG}" ]]; then
  CMD+=(--proxy-log "${PROXY_LOG}")
fi

if [[ -n "${JSON_OUT}" ]]; then
  CMD+=(--json-out "${JSON_OUT}")
fi

echo "Running (cwd=${ROOT_DIR}): ${CMD[*]}"
"${CMD[@]}"
