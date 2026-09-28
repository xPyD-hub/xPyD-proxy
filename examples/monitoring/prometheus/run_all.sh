#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec "${PYTHON:-python3}" "${SCRIPT_DIR}/run.py" --help
fi
PROMETHEUS_DIR="$(bash "${SCRIPT_DIR}/install_prometheus.sh")"
export PROMETHEUS_DIR
exec "${PYTHON:-python3}" "${SCRIPT_DIR}/run.py" "$@"
