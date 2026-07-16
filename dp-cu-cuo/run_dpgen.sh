#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m json.tool param.json >/dev/null
"${PYTHON_BIN}" -m json.tool machine.json >/dev/null
dpgen run param.json machine.json
