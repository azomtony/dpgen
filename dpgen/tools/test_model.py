import argparse
import csv
import glob
import json
import os
import re
import shlex
import subprocess
from pathlib import Path


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
DETAIL_BACKWARD_FILES = [
    "dp_test.log",
    "detail.e.out",
    "detail.e_peratom.out",
    "detail.f.out",
    "detail.v.out",
]


def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "system"


def _model_link_name(model):
    suffix = Path(model).suffix
    return f"model{suffix}" if suffix else "model"


def _load_system_names(param_file):
    if param_file is None:
        return {}
    with open(param_file) as fp:
        jdata = json.load(fp)
    names = {}
    for idx, system in enumerate(jdata.get("sys_configs", [])):
        names[f"sys.{idx:03d}"] = str(system)
    for idx, system in enumerate(jdata.get("init_data_sys", [])):
        names[f"init.{idx:03d}"] = str(system)
    return names


def _load_machine_file(filename):
    filename = str(filename)
    if filename.endswith(".json"):
        with open(filename) as fp:
            return json.load(fp)
    if filename.endswith((".yaml", ".yml")):
        from ruamel.yaml import YAML

        yaml = YAML(typ="safe", pure=True)
        with open(filename) as fp:
            return yaml.load(fp)
    raise ValueError(f"Unsupported file format: {filename}")


def _has_deepmd_system(path):
    if path.is_file():
        return path.suffix in {".h5", ".hdf5"}
    if (path / "type.raw").is_file():
        return True
    return any(path.rglob("type.raw"))


def discover_systems(systems_dir, system_prefixes=None):
    systems_dir = Path(systems_dir).resolve()
    if not systems_dir.exists():
        raise FileNotFoundError(f"{systems_dir} does not exist.")
    if _has_deepmd_system(systems_dir):
        children = sorted(
            ii
            for ii in systems_dir.iterdir()
            if not ii.name.startswith(".")
            and (
                not system_prefixes
                or any(ii.name.startswith(prefix) for prefix in system_prefixes)
            )
            and _has_deepmd_system(ii)
        )
        if children:
            return children
        return [systems_dir]
    raise RuntimeError(f"{systems_dir} does not contain any DeepMD systems.")


def discover_models(model_path, model_pattern):
    model_path = Path(model_path).resolve()
    if model_path.is_file():
        return [model_path]
    if not model_path.is_dir():
        raise FileNotFoundError(f"{model_path} does not exist.")

    models = sorted(
        Path(ii).resolve() for ii in glob.glob(str(model_path / model_pattern))
    )
    if not models and model_pattern == "graph*.pb":
        models = sorted(
            Path(ii).resolve()
            for ii in glob.glob(str(model_path / "frozen_model*.pb"))
        )
    if not models:
        raise RuntimeError(f"No models matching {model_pattern!r} found in {model_path}.")
    return models


def parse_dp_test_output(text):
    result = {
        "nframes": None,
        "energy_rmse_mev_per_atom": None,
        "force_rmse_mev_per_angstrom": None,
    }
    for line in text.splitlines():
        if re.search(r"number\s+of\s+test\s+data", line, re.IGNORECASE):
            match = re.search(r"(\d+)", line)
            if match:
                result["nframes"] = int(match.group(1))
        if re.search(r"energy\s+rmse\s*/\s*natoms", line, re.IGNORECASE):
            match = re.search(FLOAT_RE, line)
            if match:
                result["energy_rmse_mev_per_atom"] = float(match.group(0)) * 1000.0
        if (
            re.search(r"force\s+rmse", line, re.IGNORECASE)
            and not re.search(r"virial|natoms", line, re.IGNORECASE)
        ):
            match = re.search(FLOAT_RE, line)
            if match:
                result["force_rmse_mev_per_angstrom"] = float(match.group(0)) * 1000.0
    return result


def _format_value(value):
    if value in (None, ""):
        return "NA"
    return f"{value:.6g}"


def write_summary(rows, output_dir):
    output_dir = Path(output_dir)
    fields = [
        "system",
        "model",
        "nframes",
        "energy_rmse_mev_per_atom",
        "force_rmse_mev_per_angstrom",
        "status",
        "output_dir",
    ]
    with open(output_dir / "summary.tsv", "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    with open(output_dir / "summary.md", "w") as fp:
        fp.write(
            "| system | model | nframes | energy RMSE (meV/atom) | force RMSE (meV/Angstrom) | status |\n"
        )
        fp.write("| --- | --- | ---: | ---: | ---: | --- |\n")
        for row in rows:
            fp.write(
                "| {system} | {model} | {nframes} | {energy} | {force} | {status} |\n".format(
                    system=row["system"],
                    model=row["model"],
                    nframes=row["nframes"] if row["nframes"] is not None else "NA",
                    energy=_format_value(row["energy_rmse_mev_per_atom"]),
                    force=_format_value(row["force_rmse_mev_per_angstrom"]),
                    status=row["status"],
                )
            )


def write_summary_from_logs(output_dir):
    output_dir = Path(output_dir).resolve()
    summary_file = output_dir / "summary.tsv"
    if not summary_file.is_file():
        raise FileNotFoundError(f"{summary_file} does not exist.")

    rows = []
    with open(summary_file, newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            row["nframes"] = None
            row["energy_rmse_mev_per_atom"] = None
            row["force_rmse_mev_per_angstrom"] = None
            log_file = Path(row["output_dir"]) / "dp_test.log"
            if log_file.is_file():
                metrics = parse_dp_test_output(log_file.read_text())
                row["nframes"] = metrics["nframes"]
                row["energy_rmse_mev_per_atom"] = metrics["energy_rmse_mev_per_atom"]
                row["force_rmse_mev_per_angstrom"] = metrics[
                    "force_rmse_mev_per_angstrom"
                ]
                row["status"] = (
                    "ok"
                    if all(
                        row[key] is not None
                        for key in (
                            "nframes",
                            "energy_rmse_mev_per_atom",
                            "force_rmse_mev_per_angstrom",
                        )
                    )
                    else "unknown"
                )
            rows.append(row)
    write_summary(rows, output_dir)
    print_summary(rows)
    return rows


def print_summary(rows):
    headers = [
        "system",
        "model",
        "nframes",
        "energy RMSE (meV/atom)",
        "force RMSE (meV/Angstrom)",
        "status",
    ]
    table = [
        [
            row["system"],
            row["model"],
            str(row["nframes"] if row["nframes"] is not None else "NA"),
            _format_value(row["energy_rmse_mev_per_atom"]),
            _format_value(row["force_rmse_mev_per_angstrom"]),
            row["status"],
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[ii]), *(len(row[ii]) for row in table))
        for ii in range(len(headers))
    ]
    print("  ".join(headers[ii].ljust(widths[ii]) for ii in range(len(headers))))
    print("  ".join("-" * widths[ii] for ii in range(len(headers))))
    for row in table:
        print("  ".join(row[ii].ljust(widths[ii]) for ii in range(len(headers))))


def _symlink_force(src, dst):
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(Path(src).resolve(), target_is_directory=Path(src).is_dir())


def _prepare_test_cases(systems, models, output_dir, name_map):
    cases = []
    for system in systems:
        system_key = system.name
        system_name = name_map.get(system_key, system_key)
        for model in models:
            if len(models) == 1:
                run_task = _safe_name(system_key)
            else:
                run_task = os.path.join(_safe_name(system_key), _safe_name(model.stem))
            run_dir = Path(output_dir) / run_task
            cases.append(
                {
                    "system": system,
                    "system_key": system_key,
                    "system_name": system_name,
                    "model": model,
                    "run_task": run_task,
                    "run_dir": run_dir,
                }
            )
    return cases


def _get_task_mdata(machine_file, task_type):
    from dpgen.remote.decide_machine import convert_mdata

    mdata = convert_mdata(_load_machine_file(machine_file), task_types=[task_type])
    command = mdata.get(f"{task_type}_command")
    machine = mdata.get(f"{task_type}_machine")
    resources = mdata.get(f"{task_type}_resources")
    group_size = mdata.get(f"{task_type}_group_size", 1)
    if command is None or machine is None or resources is None:
        raise RuntimeError(
            f"{machine_file} must define command, machine, and resources for {task_type!r}."
        )
    return mdata, command, machine, resources, group_size


def _check_api_version(mdata):
    version = str(mdata.get("api_version", "1.0"))
    parts = []
    for item in version.split(".")[:2]:
        try:
            parts.append(int(item))
        except ValueError:
            parts.append(0)
    while len(parts) < 2:
        parts.append(0)
    if tuple(parts) < (1, 0):
        raise RuntimeError(
            "API version below 1.0 is no longer supported. Please upgrade to version 1.0 or newer."
        )


def _make_submission(*args, **kwargs):
    from dpgen.dispatcher.Dispatcher import make_submission

    return make_submission(*args, **kwargs)


def run_model_tests(
    systems_dir,
    model_path,
    output_dir,
    dp_command=None,
    model_pattern="graph*.pb",
    numb_test=0,
    param_file=None,
    keep_going=False,
    extra_dp_args=None,
    system_prefixes=None,
    machine_file=None,
    task_type="train",
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    systems = discover_systems(systems_dir, system_prefixes=system_prefixes)
    models = discover_models(model_path, model_pattern)
    name_map = _load_system_names(param_file)
    extra_dp_args = extra_dp_args or []
    rows = []
    cases = _prepare_test_cases(systems, models, output_dir, name_map)

    if machine_file is not None:
        submit_model_tests(
            cases,
            output_dir,
            machine_file,
            task_type,
            dp_command=dp_command,
            numb_test=numb_test,
            extra_dp_args=extra_dp_args,
            keep_going=keep_going,
        )
        return write_summary_from_logs(output_dir)

    command_prefix = shlex.split(dp_command or "dp")
    for case in cases:
        run_dir = case["run_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)

        detail_prefix = run_dir / "detail"
        log_file = run_dir / "dp_test.log"
        command = [
            *command_prefix,
            "test",
            "-m",
            str(case["model"]),
            "-s",
            str(case["system"]),
            "-n",
            str(numb_test),
            "-d",
            str(detail_prefix),
            *extra_dp_args,
        ]
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log_file.write_text(proc.stdout)
        metrics = parse_dp_test_output(proc.stdout)
        status = "ok" if proc.returncode == 0 else f"failed:{proc.returncode}"
        rows.append(
            {
                "system": case["system_name"],
                "model": case["model"].name,
                "nframes": metrics["nframes"],
                "energy_rmse_mev_per_atom": metrics["energy_rmse_mev_per_atom"],
                "force_rmse_mev_per_angstrom": metrics[
                    "force_rmse_mev_per_angstrom"
                ],
                "status": status,
                "output_dir": str(run_dir),
            }
        )
        write_summary(rows, output_dir)
        if proc.returncode != 0 and not keep_going:
            raise RuntimeError(
                f"`dp test` failed for {case['system_name']} with {case['model'].name}. See {log_file}."
            )
    print_summary(rows)
    return rows


def submit_model_tests(
    cases,
    output_dir,
    machine_file,
    task_type,
    dp_command=None,
    numb_test=0,
    extra_dp_args=None,
    keep_going=False,
):
    mdata, command, machine, resources, group_size = _get_task_mdata(
        machine_file, task_type
    )
    _check_api_version(mdata)
    if dp_command is not None:
        command = dp_command

    output_dir = Path(output_dir).resolve()
    extra_dp_args = extra_dp_args or []
    rows = []
    forward_files = []
    backward_files = ["batch.log"]
    commands = []
    for case in cases:
        run_dir = case["run_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)
        _symlink_force(case["system"], run_dir / "system")
        model_link_name = _model_link_name(case["model"])
        _symlink_force(case["model"], run_dir / model_link_name)
        forward_files.extend(
            [
                os.path.join(case["run_task"], "system"),
                os.path.join(case["run_task"], model_link_name),
            ]
        )
        backward_files.extend(
            os.path.join(case["run_task"], item)
            for item in DETAIL_BACKWARD_FILES
        )
        rows.append(
            {
                "system": case["system_name"],
                "model": case["model"].name,
                "nframes": None,
                "energy_rmse_mev_per_atom": None,
                "force_rmse_mev_per_angstrom": None,
                "status": "submitted",
                "output_dir": str(run_dir),
            }
        )

        test_command = [
            *shlex.split(command),
            "test",
            "-m",
            model_link_name,
            "-s",
            "system",
            "-n",
            str(numb_test),
            "-d",
            "detail",
            *extra_dp_args,
        ]
        one_test = " ".join(shlex.quote(part) for part in test_command)
        redirect = f"{one_test} > dp_test.log 2>&1"
        task_command = f"( cd {shlex.quote(case['run_task'])} && {redirect} )"
        if keep_going:
            task_command = f"{task_command} || true"
        commands.append(task_command)
    write_summary(rows, output_dir)
    submission = _make_submission(
        machine,
        resources,
        commands=[" && ".join(commands)],
        work_path=str(output_dir),
        run_tasks=["."],
        group_size=group_size,
        forward_common_files=[],
        forward_files=sorted(set(forward_files)),
        backward_files=sorted(set(backward_files)),
        outlog="batch.log",
        errlog="batch.log",
    )
    submission.run_submission()


def gen_test_model(args):
    extra_dp_args = shlex.split(args.extra_dp_args)
    run_model_tests(
        args.SYSTEMS_DIR,
        args.MODEL,
        args.OUTPUT,
        dp_command=args.dp_command,
        model_pattern=args.model_pattern,
        numb_test=args.numb_test,
        param_file=args.param,
        keep_going=args.keep_going,
        extra_dp_args=extra_dp_args,
        system_prefixes=args.system_prefix,
        machine_file=args.machine,
        task_type=args.task_type,
    )


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "test-model",
        help="Test DeePMD models on each system in a collected data directory.",
    )
    parser.add_argument("SYSTEMS_DIR", type=str, help="directory containing systems")
    parser.add_argument("MODEL", type=str, help="model file or directory of models")
    parser.add_argument("OUTPUT", type=str, help="directory for test outputs")
    parser.add_argument(
        "-p",
        "--param",
        type=str,
        default=None,
        help="optional param.json for mapping sys.NNN names to original system names",
    )
    parser.add_argument(
        "--model-pattern",
        type=str,
        default="graph*.pb",
        help="glob pattern used when MODEL is a directory",
    )
    parser.add_argument(
        "-n",
        "--numb-test",
        type=int,
        default=0,
        help="number of frames tested per system; 0 means all frames",
    )
    parser.add_argument(
        "--dp-command",
        type=str,
        default=None,
        help="DeePMD-kit command to run; defaults to dp locally or machine task command when submitted",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue testing other systems if one dp test command fails",
    )
    parser.add_argument(
        "--machine",
        type=str,
        default=None,
        help="machine.json/yaml used to submit dp test tasks with DPDispatcher",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="train",
        help="machine-file section to use for submitted tests",
    )
    parser.add_argument(
        "--system-prefix",
        action="append",
        default=None,
        help="only test entries whose directory name starts with this prefix; can be repeated",
    )
    parser.add_argument(
        "--extra-dp-args",
        type=str,
        default="",
        help='extra arguments passed to dp test, for example "--head energy"',
    )
    parser.set_defaults(func=gen_test_model)
    return parser


def _main():
    parser = argparse.ArgumentParser(
        description="Test DeePMD models on each system in a collected data directory."
    )
    parser.add_argument("SYSTEMS_DIR", type=str, help="directory containing systems")
    parser.add_argument("MODEL", type=str, help="model file or directory of models")
    parser.add_argument("OUTPUT", type=str, help="directory for test outputs")
    parser.add_argument("-p", "--param", type=str, default=None)
    parser.add_argument("--model-pattern", type=str, default="graph*.pb")
    parser.add_argument("-n", "--numb-test", type=int, default=0)
    parser.add_argument("--dp-command", type=str, default=None)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--system-prefix", action="append", default=None)
    parser.add_argument("--machine", type=str, default=None)
    parser.add_argument("--task-type", type=str, default="train")
    parser.add_argument("--extra-dp-args", type=str, default="")
    parser.set_defaults(func=gen_test_model)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    _main()
