#!/usr/bin/env python3
"""Render a Cu/CuO DP-GEN param.json from a compact project JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render DP-GEN param.json for Cu/CuO.")
    parser.add_argument("project", help="Project JSON, usually configs/project.json.")
    parser.add_argument("output", help="Output param.json path.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        return json.load(fp)


def default_training_param(type_map: list[str], stop_batch: int) -> dict[str, Any]:
    return {
        "model": {
            "type_map": type_map,
            "descriptor": {
                "type": "se_e2_a",
                "sel": [120 for _ in type_map],
                "rcut_smth": 0.5,
                "rcut": 6.0,
                "neuron": [120, 120, 120],
                "resnet_dt": True,
                "axis_neuron": 16,
                "seed": 1,
            },
            "fitting_net": {
                "neuron": [240, 240, 240],
                "resnet_dt": True,
                "seed": 1,
            },
        },
        "learning_rate": {
            "type": "exp",
            "start_lr": 0.001,
            "decay_steps": 5000,
            "stop_lr": 1.0e-8,
        },
        "loss": {
            "start_pref_e": 0.02,
            "limit_pref_e": 1.0,
            "start_pref_f": 1000,
            "limit_pref_f": 1.0,
            "start_pref_v": 0.0,
            "limit_pref_v": 0.0,
        },
        "training": {
            "stop_batch": stop_batch,
            "seed": 1,
            "disp_file": "lcurve.out",
            "disp_freq": 1000,
            "save_freq": 1000,
            "save_ckpt": "model.ckpt",
            "disp_training": True,
            "time_training": True,
            "profiling": False,
            "training_data": {
                "systems": [],
                "batch_size": "auto",
            },
        },
    }


def build_param(project: dict[str, Any]) -> dict[str, Any]:
    type_map = project.get("type_map", ["Cu", "O"])
    stop_batch = int(project.get("training_stop_batch", 200000))

    param = {
        "type_map": type_map,
        "mass_map": project.get("mass_map", [63.546, 15.999]),
        "init_data_prefix": project["init_data_prefix"],
        "init_data_sys": project["init_data_sys"],
        "sys_configs_prefix": project["sys_configs_prefix"],
        "sys_configs": project["sys_configs_globs"],
        "numb_models": int(project.get("numb_models", 4)),
        "train_backend": project.get("train_backend", "pytorch"),
        "default_training_param": project.get(
            "default_training_param",
            default_training_param(type_map, stop_batch),
        ),
        "model_devi_dt": float(project.get("model_devi_dt", 0.001)),
        "model_devi_skip": int(project.get("model_devi_skip", 0)),
        "model_devi_f_trust_lo": float(project.get("model_devi_f_trust_lo", 0.08)),
        "model_devi_f_trust_hi": float(project.get("model_devi_f_trust_hi", 0.25)),
        "model_devi_clean_traj": bool(project.get("model_devi_clean_traj", True)),
        "model_devi_jobs": project["model_devi_jobs"],
        "fp_style": "vasp",
        "shuffle_poscar": bool(project.get("shuffle_poscar", False)),
        "fp_task_max": int(project.get("fp_task_max", 50)),
        "fp_task_min": int(project.get("fp_task_min", 5)),
        "fp_pp_path": project["fp_pp_path"],
        "fp_pp_files": project["fp_pp_files"],
        "fp_incar": project["fp_incar"],
    }

    if "sys_batch_size" in project:
        param["sys_batch_size"] = project["sys_batch_size"]
    if "init_batch_size" in project:
        param["init_batch_size"] = project["init_batch_size"]
    if "fp_kpoints" in project:
        param["fp_kpoints"] = project["fp_kpoints"]

    return param


def main() -> None:
    args = parse_args()
    project = load_json(Path(args.project))
    param = build_param(project)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fp:
        json.dump(param, fp, indent=2)
        fp.write("\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
