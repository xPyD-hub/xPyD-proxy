#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python3}" -c '
import sys
# Override the caller directory that Python places ahead of PYTHONPATH.
sys.path.insert(0, sys.argv.pop(1))
from xpyd.proxy import main
main()
' "${REPO_ROOT}" "$@"
