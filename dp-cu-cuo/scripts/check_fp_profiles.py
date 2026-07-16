#!/usr/bin/env python3
"""Check task-specific INCAR and KPOINTS files after DP-GEN make_fp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate per-system INCAR/KPOINTS files in iter.XXXXXX/02.fp/task.*."
    )
    parser.add_argument("param", help="DP-GEN param.json.")
    parser.add_argument("fp_dir", help="FP directory, e.g. iter.000000/02.fp.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        return json.load(fp)


def task_sys_idx(task: Path) -> str:
    parts = task.name.split(".")
    if len(parts) < 3 or parts[0] != "task":
        raise ValueError(f"Unexpected task name: {task.name}")
    return str(int(parts[1]))


def select_by_sys(value: Any, sys_idx: str) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        idx = int(sys_idx)
        if idx >= len(value):
            raise KeyError(f"No list entry for system index {sys_idx}")
        return value[idx]
    if isinstance(value, dict):
        if sys_idx in value:
            return value[sys_idx]
        return value.get("default")
    return None


def format_kpoints(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        style = "Monkhorst-Pack"
        mesh = value
        shift = [0, 0, 0]
    elif isinstance(value, dict):
        style = value.get("style", "Monkhorst-Pack")
        mesh = value["mesh"]
        shift = value.get("shift", [0, 0, 0])
    else:
        raise TypeError("KPOINTS profile must be a mesh list or object")
    mesh = [int(item) for item in mesh]
    shift = [float(item) for item in shift]
    return "\n".join(
        [
            "Automatic mesh",
            "0",
            style,
            "{} {} {}".format(*mesh),
            "{:g} {:g} {:g}".format(*shift),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    param_path = Path(args.param).resolve()
    root = param_path.parent
    fp_dir = Path(args.fp_dir)
    param = load_json(param_path)

    tasks = sorted(path for path in fp_dir.glob("task.*") if path.is_dir())
    if not tasks:
        raise SystemExit(f"No task.* directories found under {fp_dir}")

    errors: list[str] = []
    checked_incar = 0
    checked_kpoints = 0

    for task in tasks:
        sys_idx = task_sys_idx(task)

        incar_profile = select_by_sys(param.get("fp_incar"), sys_idx)
        if incar_profile is not None:
            expected_path = root / incar_profile
            actual_path = task / "INCAR"
            if not actual_path.is_file():
                errors.append(f"{task}: missing INCAR")
            elif expected_path.read_text().strip() != actual_path.read_text().strip():
                errors.append(f"{task}: INCAR does not match {incar_profile}")
            checked_incar += 1

        kpoints_profile = select_by_sys(param.get("fp_kpoints"), sys_idx)
        expected_kpoints = format_kpoints(kpoints_profile)
        if expected_kpoints is not None:
            actual_path = task / "KPOINTS"
            if not actual_path.is_file():
                errors.append(f"{task}: missing KPOINTS")
            elif expected_kpoints.strip() != actual_path.read_text().strip():
                errors.append(f"{task}: KPOINTS does not match fp_kpoints for system {sys_idx}")
            checked_kpoints += 1

    if errors:
        print("FP profile check failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print(
        f"FP profile check passed for {len(tasks)} tasks "
        f"({checked_incar} INCAR checks, {checked_kpoints} explicit KPOINTS checks)."
    )


if __name__ == "__main__":
    main()
