#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m json.tool param.json >/dev/null
"${PYTHON_BIN}" -m json.tool machine.train.json >/dev/null
"${PYTHON_BIN}" scripts/validate_dpgen_inputs.py --train-only param.json machine.train.json
"${PYTHON_BIN}" scripts/run_initial_train.py param.json machine.train.json
