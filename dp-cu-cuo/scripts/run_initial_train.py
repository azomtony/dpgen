#!/usr/bin/env python3
"""Run only DP-GEN initial training stages for iter.000000."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dpgen import dlog
from dpgen.generator.arginfo import run_jdata_arginfo
from dpgen.generator.lib.utils import record_iter
from dpgen.generator.run import make_train, post_train, run_train, update_mass_map
from dpgen.remote.decide_machine import convert_mdata
from dpgen.util import load_file, normalize, setup_ele_temp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build, submit, and post-process DP-GEN iter.000000/00.train only. "
            "Use this when initial labeled data already exists."
        )
    )
    parser.add_argument("param", help="DP-GEN param.json.")
    parser.add_argument("machine", help="Train-only machine JSON.")
    parser.add_argument("--debug", action="store_true", help="Enable DP-GEN debug logging.")
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Do not append 0 0, 0 1, 0 2 to record.dpgen.",
    )
    return parser.parse_args()


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

    jdata = prepare_jdata(str(param_path))
    mdata = convert_mdata(load_file(str(machine_path)))

    record = "record.dpgen"
    dlog.info("initial train: make_train")
    make_train(0, jdata, mdata)
    if not args.no_record:
        record_iter(record, 0, 0)

    dlog.info("initial train: run_train")
    run_train(0, jdata, mdata)
    if not args.no_record:
        record_iter(record, 0, 1)

    dlog.info("initial train: post_train")
    post_train(0, jdata, mdata)
    if not args.no_record:
        record_iter(record, 0, 2)

    print("Initial training stage complete through iter.000000 task 02.")


if __name__ == "__main__":
    main()
