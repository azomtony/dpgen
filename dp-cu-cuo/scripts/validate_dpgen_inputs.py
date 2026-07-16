#!/usr/bin/env python3
"""Validate common Cu/CuO DP-GEN input mistakes before running on HPC."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate param.json and machine.json.")
    parser.add_argument("param", help="DP-GEN param.json.")
    parser.add_argument("machine", help="DP-GEN machine.json.")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only require the train machine stage and initial training inputs.",
    )
    parser.add_argument(
        "--no-fp-machine",
        action="store_true",
        help="Require train/model_devi stages but do not require an fp machine stage.",
    )
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        return json.load(fp)


def check_path(path: Path, label: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"{label} does not exist: {path}")


def contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return "YOUR_" in value or "/path/to/" in value or value.endswith("_QUEUE")
    if isinstance(value, dict):
        return any(contains_placeholder(v) for v in value.values())
    if isinstance(value, list):
        return any(contains_placeholder(v) for v in value)
    return False


def iter_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, str)]
    return []


def main() -> None:
    args = parse_args()
    param_path = Path(args.param).resolve()
    machine_path = Path(args.machine).resolve()
    root = param_path.parent
    param = load_json(param_path)
    machine = load_json(machine_path)

    errors: list[str] = []
    warnings: list[str] = []

    type_map = param.get("type_map")
    if type_map != ["Cu", "O"]:
        warnings.append(f"Expected type_map ['Cu', 'O']; got {type_map!r}")
    if len(param.get("mass_map", [])) != len(type_map or []):
        errors.append("mass_map length must match type_map length.")

    init_prefix = root / param.get("init_data_prefix", "")
    check_path(init_prefix, "init_data_prefix", errors)
    for rel in param.get("init_data_sys", []):
        check_path(init_prefix / rel, f"init_data_sys entry {rel}", errors)

    if not args.train_only:
        sys_prefix = root / param.get("sys_configs_prefix", "")
        check_path(sys_prefix, "sys_configs_prefix", errors)
        for idx, group in enumerate(param.get("sys_configs", [])):
            matches = []
            for pattern in group:
                matches.extend(glob.glob(str(sys_prefix / pattern), recursive=True))
            if not matches:
                errors.append(f"sys_configs[{idx}] matched no POSCAR files.")
            else:
                print(f"sys_configs[{idx}] matches {len(matches)} files.")

        pp_path = root / param.get("fp_pp_path", "")
        check_path(pp_path, "fp_pp_path", errors)
        for pp_file in param.get("fp_pp_files", []):
            check_path(pp_path / pp_file, f"fp_pp_file {pp_file}", errors)

        for fp_incar in iter_paths(param.get("fp_incar", "")):
            check_path(root / fp_incar, f"fp_incar {fp_incar}", errors)

    if args.train_only:
        stages = ("train",)
    elif args.no_fp_machine:
        stages = ("train", "model_devi")
    else:
        stages = ("train", "model_devi", "fp")
    for stage in stages:
        if stage not in machine:
            errors.append(f"machine.json missing stage: {stage}")
            continue
        if "command" not in machine[stage]:
            errors.append(f"machine.json {stage} missing command.")
        if contains_placeholder(machine[stage]):
            warnings.append(f"machine.json {stage} still contains placeholders.")
        resources = machine[stage].get("resources", {})
        if not resources.get("source_list"):
            warnings.append(f"machine.json {stage} has no resources.source_list activation.")

    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f"  - {item}")

    if errors:
        print("\nErrors:")
        for item in errors:
            print(f"  - {item}")
        raise SystemExit(1)

    if args.strict and warnings:
        raise SystemExit(1)

    print("\nValidation passed.")


if __name__ == "__main__":
    main()
