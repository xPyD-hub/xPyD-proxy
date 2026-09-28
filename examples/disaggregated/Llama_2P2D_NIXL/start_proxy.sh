#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../../lib/run_proxy.sh" proxy \
    -c "${SCRIPT_DIR}/xpyd_2p2d.yaml"
