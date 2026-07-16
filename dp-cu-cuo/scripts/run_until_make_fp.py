#!/usr/bin/env python3
"""Run DP-GEN stages through make_fp, then stop before VASP submission."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dpgen import dlog
from dpgen.generator.arginfo import run_jdata_arginfo
from dpgen.generator.lib.utils import make_iter_name, record_iter
from dpgen.generator.run import (
    make_fp,
    make_model_devi,
    make_train,
    post_model_devi,
    post_train,
    run_model_devi,
    run_train,
    update_mass_map,
)
from dpgen.remote.decide_machine import convert_mdata
from dpgen.util import load_file, normalize, sepline, setup_ele_temp


TASKS = [
    ("make_train", make_train),
    ("run_train", run_train),
    ("post_train", post_train),
    ("make_model_devi", make_model_devi),
    ("run_model_devi", run_model_devi),
    ("post_model_devi", post_model_devi),
    ("make_fp", make_fp),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one DP-GEN iteration through task 06 make_fp, then stop before "
            "task 07 run_fp so VASP jobs are not submitted."
        )
    )
    parser.add_argument("param", help="DP-GEN param.json.")
    parser.add_argument("machine", help="Machine JSON with train and model_devi stages.")
    parser.add_argument(
        "--iter-index",
        type=int,
        default=None,
        help="Iteration index. Defaults to the iteration in record.dpgen, or 0.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DP-GEN debug logging.")
    return parser.parse_args()


def read_record(record: Path) -> tuple[int, int]:
    if not record.is_file():
        return 0, -1
    last = None
    with record.open() as fp:
        for line in fp:
            stripped = line.strip()
            if stripped:
                last = stripped
    if last is None:
        return 0, -1
    ii, jj = last.split()[:2]
    return int(ii), int(jj)


def prepare_jdata(param_file: str) -> dict:
    jdata = load_file(param_file)
    jdata = normalize(run_jdata_arginfo(), jdata, strict_check=False)
    update_mass_map(jdata)
    use_ele_temp = jdata.get("use_ele_temp", 0)
    if use_ele_temp == 1:
        setup_ele_temp(False)
    elif use_ele_temp == 2:
        setup_ele_temp(True)
    return jdata


def main() -> None:
    args = parse_args()
    if args.debug:
        dlog.setLevel(logging.DEBUG)

    param_path = Path(args.param)
    machine_path = Path(args.machine)
    if not param_path.is_file():
        raise SystemExit(f"Missing param file: {param_path}")
    if not machine_path.is_file():
        raise SystemExit(f"Missing machine file: {machine_path}")

    record = Path("record.dpgen")
    rec_iter, rec_task = read_record(record)
    iter_index = rec_iter if args.iter_index is None else args.iter_index

    jdata = prepare_jdata(str(param_path))
    mdata = convert_mdata(load_file(str(machine_path)), task_types=["train", "model_devi"])

    sepline(make_iter_name(iter_index), "=")
    for task_index, (task_name, task_fn) in enumerate(TASKS):
        if iter_index == rec_iter and task_index <= rec_task:
            dlog.info(
                "skip recorded stage iter %06d task %02d %s",
                iter_index,
                task_index,
                task_name,
            )
            continue

        sepline(f"{make_iter_name(iter_index)} task {task_index:02d} {task_name}", "-")
        result = task_fn(iter_index, jdata, mdata)
        record_iter(str(record), iter_index, task_index)
        if task_name == "make_model_devi" and result is False:
            print("No more model_devi jobs for this iteration.")
            return

    print(f"Stopped after {make_iter_name(iter_index)} task 06 make_fp.")
    print("Review or overwrite INCAR and KPOINTS in 02.fp/task.* before VASP.")


if __name__ == "__main__":
    main()
