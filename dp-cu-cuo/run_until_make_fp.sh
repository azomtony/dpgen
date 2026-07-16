#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m json.tool param.json >/dev/null
"${PYTHON_BIN}" -m json.tool machine.json >/dev/null
"${PYTHON_BIN}" scripts/validate_dpgen_inputs.py --no-fp-machine param.json machine.json
"${PYTHON_BIN}" scripts/run_until_make_fp.py param.json machine.json
