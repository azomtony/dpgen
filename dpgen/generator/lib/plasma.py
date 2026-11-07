"""Task generation, continuation, and event-balanced plasma labeling."""

import copy
import csv
import itertools
import json
import math
import os
import random
import secrets
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import dpdata
import numpy as np

from dpgen import dlog
from dpgen.generator.lib import plasma_runtime
from dpgen.generator.lib.utils import create_path, make_iter_name

DEFAULTS = {
    "nions": 10,
    "eion": 10.0,
    "ratio": 10,
    "ion_dt": 5.0,
    "surface": 100,
    "a": 3.62716774,
    "eq_time": 5.0,
    "eq_trj_freq": 1000,
    "checkpoint_freq": 1000,
    "edge": 2.0,
    "margin": 2.0,
    "height": 8.0,
    "initial_dt": 0.00001,
    "max_dt": 0.0005,
    "max_displacement": 0.01,
}
LEGACY_SELECTION = {
    "eq_task_max": 5,
    "ion_task_max": 30,
    "neutral_task_max": 20,
    "validation_task_max": 5,
    "max_frames_per_event": 3,
    "min_time_separation": 0.01,
}
SELECTION = {
    "eq_fraction": 0.1,
    "ion_fraction": 0.3,
    "neutral_fraction": 0.5,
    "validation_fraction": 0.1,
    "max_frames_per_event": 3,
    "min_time_separation": 0.01,
}
MATCH_FIELDS = (
    "eion",
    "ratio",
    "ion_dt",
    "surface",
    "a",
    "edge",
    "margin",
    "height",
    "eq_time",
)


def limit(value, system):
    if isinstance(value, dict):
        return float(value[str(system)])
    if isinstance(value, list):
        return float(value[system])
    return float(value)


def settings(job, system_position=0):
    options = dict(job.get("plasma", {}))
    surface = options.get("surface", DEFAULTS["surface"])
    if isinstance(surface, list):
        if len(surface) != len(job["sys_idx"]):
            raise ValueError(
                "plasma.surface list must have one entry per sys_idx, in the same order"
            )
        options["surface"] = surface[system_position]
    unknown = options.keys() - (DEFAULTS.keys() | {"seed"})
    if unknown:
        raise ValueError(f"Unknown plasma settings: {sorted(unknown)}")
    if isinstance(options.get("seed"), list):
        raise ValueError("plasma.seed must be a scalar integer")
    if any(isinstance(value, list) and not value for value in options.values()):
        raise ValueError("Plasma sweep lists must not be empty")
    base = dict(DEFAULTS, **options)
    keys = list(base)
    variants = itertools.product(
        *(value if isinstance(value, list) else [value] for value in base.values())
    )
    for values in variants:
        config = dict(zip(keys, values))
        for key in DEFAULTS.keys() | {"seed"}:
            if key not in config:
                continue
            value = config[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"plasma.{key} must be a finite number")
            if value < 0 or (
                key not in {"ratio", "eq_time", "edge", "margin"} and value == 0
            ):
                raise ValueError(f"Invalid plasma.{key}: {value}")
        for key in (
            "nions",
            "ratio",
            "surface",
            "eq_trj_freq",
            "checkpoint_freq",
            "seed",
        ):
            if key in config and int(config[key]) != config[key]:
                raise ValueError(f"plasma.{key} must be an integer")
            if key in config:
                config[key] = int(config[key])
        if config["surface"] not in {100, 110, 111}:
            raise ValueError("plasma.surface must be 100, 110, or 111")
        if config["initial_dt"] > config["max_dt"]:
            raise ValueError("plasma.initial_dt must not exceed max_dt")
        yield config


def selection_options(job):
    supplied = job.get("selection", {})
    legacy = any(key.endswith("_task_max") for key in supplied)
    if legacy and any(key.endswith("_fraction") for key in supplied):
        raise ValueError(
            "Use fractional or absolute plasma selection budgets, not both"
        )
    defaults = LEGACY_SELECTION if legacy else SELECTION
    value = dict(defaults, **supplied)
    if value.keys() - defaults.keys():
        raise ValueError("Unknown plasma selection option")
    for key, number in value.items():
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number < 0
        ):
            raise ValueError(f"Invalid plasma selection.{key}")
        if key.endswith("_fraction"):
            if number > 1:
                raise ValueError(f"selection.{key} must be in [0, 1]")
        elif key != "min_time_separation" and int(number) != number:
            raise ValueError(f"selection.{key} must be an integer")
    if not legacy and not math.isclose(
        sum(value[key] for key in value if key.endswith("_fraction")),
        1.0,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Plasma selection fractions must sum to 1, including validation_fraction"
        )
    if value["max_frames_per_event"] < 1:
        raise ValueError("selection.max_frames_per_event must be positive")
    value["_legacy"] = legacy
    return value


def selection_budgets(job, total_budget):
    """Apportion the total label budget with deterministic largest remainders."""
    if (
        isinstance(total_budget, bool)
        or total_budget < 0
        or int(total_budget) != total_budget
    ):
        raise ValueError("Plasma total FP budget must be a nonnegative integer")
    options = selection_options(job)
    kinds = ("eq", "ion", "neutral", "validation")
    if options["_legacy"]:
        return options, {kind: options[f"{kind}_task_max"] for kind in kinds}
    exact = {kind: total_budget * options[f"{kind}_fraction"] for kind in kinds}
    budgets = {kind: math.floor(value) for kind, value in exact.items()}
    order = sorted(kinds, key=lambda kind: -(exact[kind] - budgets[kind]))
    for kind in order[: int(total_budget - sum(budgets.values()))]:
        budgets[kind] += 1
    return options, budgets


def _signature(system, source, temperature, config):
    return (
        system,
        os.path.realpath(source),
        temperature,
        *(config[key] for key in MATCH_FIELDS),
    )


def _previous(iter_index):
    index = defaultdict(list)
    if not iter_index:
        return index
    for task in sorted(
        (Path(make_iter_name(iter_index - 1)) / "01.model_devi").glob("task.*.*")
    ):
        if not (task / "frames.json").exists():
            continue
        job = json.loads((task / "job.json").read_text())
        if job.get("ensemble") == "plasma":
            index[
                _signature(
                    int(task.name.split(".")[1]),
                    job["source_conf"],
                    job["temps"],
                    job["plasma"],
                )
            ].append(task)
    return index


def _write_continuation(dump, destination, type_map):
    system = dpdata.System(str(dump), fmt="lammps/dump", type_map=type_map)
    lines = Path(dump).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS "))
    columns = lines[start].split()[2:]
    indexes = [columns.index(name) for name in ("id", "vx", "vy", "vz")]
    rows = [
        [float(line.split()[index]) for index in indexes]
        for line in lines[start + 1 :]
        if line.strip()
    ]
    rows.sort(key=lambda row: row[0])
    if len(rows) != system.get_natoms() or not np.isfinite(rows).all():
        raise ValueError("Invalid plasma restart velocities")
    system.to_lammps_lmp(str(destination))
    with Path(destination).open("a") as stream:
        stream.write("\nVelocities\n\n")
        for index, row in enumerate(rows, 1):
            stream.write(f"{index} {row[1]:.17g} {row[2]:.17g} {row[3]:.17g}\n")


def continuation(tasks, config, jdata, iter_index, system, destination):
    if not tasks:
        return {"source": "initial", "reason": "no matching plasma trajectory"}
    previous_job = jdata["model_devi_jobs"][iter_index - 1]
    f_hi = limit(
        previous_job.get("model_devi_f_trust_hi", jdata["model_devi_f_trust_hi"]),
        system,
    )
    v_hi = limit(
        previous_job.get(
            "model_devi_v_trust_hi", jdata.get("model_devi_v_trust_hi", 1e10)
        ),
        system,
    )
    candidates = []
    for task in tasks:
        try:
            metadata = json.loads((task / "frames.json").read_text())
            for row in np.atleast_2d(np.loadtxt(task / "model_devi.out")):
                if (
                    len(row) >= 7
                    and np.isfinite(row[[0, 1, 4]]).all()
                    and 0 <= row[1] < v_hi
                    and 0 <= row[4] < f_hi
                ):
                    step = int(row[0])
                    if (
                        step >= jdata.get("model_devi_skip", 0)
                        and str(step) in metadata
                    ):
                        candidates.append((step, task, metadata[str(step)]))
        except (OSError, ValueError, KeyError, IndexError) as exc:
            dlog.warning("Cannot read plasma continuation %s: %s", task, exc)
    for step, task, frame in sorted(
        candidates, key=lambda item: (item[0], str(item[1])), reverse=True
    ):
        try:
            _write_continuation(
                task / "traj" / f"{step}.lammpstrj", destination, jdata["type_map"]
            )
            config.update(
                eq_complete=frame["eq_complete"],
                eq_elapsed=frame["eq_elapsed"],
                cumulative_arrivals=frame["cumulative_arrivals"],
                cumulative_ions=frame["cumulative_ions"],
                cumulative_neutrals=frame["cumulative_neutrals"],
            )
            return {"source": str(task), "step": step, "frame": frame}
        except (
            OSError,
            ValueError,
            RuntimeError,
            AssertionError,
            KeyError,
            IndexError,
        ) as exc:
            dlog.warning("Skipping plasma restart %s frame %d: %s", task, step, exc)
    return {
        "source": "initial",
        "reason": "no usable accurate or candidate plasma frame",
    }


def make_tasks(iter_index, jdata, conf_systems, models):
    job = jdata["model_devi_jobs"][iter_index]
    if jdata["type_map"] != ["Cu", "O"]:
        raise ValueError("Plasma deposition requires type_map ['Cu', 'O']")
    if len(models) < 2:
        raise ValueError("Plasma exploration requires at least two committee models")
    if (
        jdata.get("model_devi_merge_traj", False)
        or jdata.get("use_clusters", False)
        or jdata.get("cluster_cutoff") is not None
    ):
        raise ValueError(
            "Plasma requires separate trajectory frames and whole-structure labeling"
        )
    if (
        jdata.get("use_ele_temp", 0)
        or jdata.get("model_devi_f_avg_relative", False)
        or jdata.get("model_devi_adapt_trust_lo", False)
        or jdata.get("use_relative", False)
        or jdata.get("use_relative_v", False)
    ):
        raise ValueError(
            "Plasma currently supports standard force/virial trust thresholds without electronic temperature"
        )
    if jdata.get("fp_style", "vasp") != "vasp":
        raise ValueError(
            "Plasma labeling currently supports the existing VASP single-point workflow"
        )
    factor = job.get("plasma_stop_factor", 2.0)
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not math.isfinite(factor)
        or factor < 1
    ):
        raise ValueError("plasma_stop_factor must be finite and at least 1")
    frequency = job["trj_freq"]
    if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency < 1:
        raise ValueError("Plasma trj_freq must be a positive integer")
    selection_options(job)
    if not job.get("temps"):
        raise ValueError("Plasma requires at least one temperature")
    configs_by_system = [
        list(settings(job, position)) for position in range(len(job["sys_idx"]))
    ]
    previous = (
        _previous(iter_index)
        if jdata.get("model_devi_restart_from_last", False)
        else {}
    )
    root = Path(make_iter_name(iter_index)) / "01.model_devi"
    for system_index, sources, configs in zip(
        job["sys_idx"], conf_systems, configs_by_system
    ):
        counter = 0
        for source, temperature, base in itertools.product(
            sources, job["temps"], configs
        ):
            if (
                not isinstance(temperature, (int, float))
                or temperature <= 0
                or not math.isfinite(temperature)
            ):
                raise ValueError("Plasma temperature must be positive and finite")
            task = root / f"task.{system_index:03d}.{counter:06d}"
            counter += 1
            create_path(str(task))
            (task / "traj").mkdir()
            config = copy.deepcopy(base)
            # Distinct reproducible seeds for a supplied base; otherwise fresh.
            base_seed = config.pop("seed", secrets.randbelow(899999999) + 1)
            config.update(
                seed=plasma_runtime.seed_for(
                    base_seed, iter_index, system_index, counter
                ),
                temp=temperature,
                trj_freq=frequency,
                stop_factor=factor,
                models=[f"../{Path(model).name}" for model in models],
                f_hi=limit(
                    job.get("model_devi_f_trust_hi", jdata["model_devi_f_trust_hi"]),
                    system_index,
                ),
                v_hi=limit(
                    job.get(
                        "model_devi_v_trust_hi",
                        jdata.get("model_devi_v_trust_hi", 1e10),
                    ),
                    system_index,
                ),
            )
            if config["f_hi"] <= 0 or config["v_hi"] <= 0:
                raise ValueError("Plasma upper trust thresholds must be positive")
            signature = _signature(system_index, source, temperature, config)
            provenance = continuation(
                previous.get(signature, []),
                config,
                jdata,
                iter_index,
                system_index,
                task / "conf.lmp",
            )
            if provenance["source"] == "initial":
                conf = dpdata.System(
                    source,
                    fmt=jdata.get("sys_format", "vasp/poscar"),
                    type_map=jdata["type_map"],
                )
                if np.any(
                    np.abs(
                        conf.data["cells"][0] - np.diag(np.diag(conf.data["cells"][0]))
                    )
                    > 1e-8
                ):
                    raise ValueError("Plasma deposition requires a rectangular cell")
                conf.to_lammps_lmp(str(task / "conf.lmp"))
            plasma_runtime.atomic_json(task / "plasma.json", config)
            plasma_runtime.atomic_json(
                task / "job.json",
                {
                    "ensemble": "plasma",
                    "temps": temperature,
                    "press": -1,
                    "source_conf": os.path.realpath(source),
                    "plasma": config,
                    "restart": provenance,
                },
            )
            shutil.copyfile(plasma_runtime.__file__, task / "plasma_runtime.py")
            (task / "checkpoints").mkdir()
            # Preview of the first block; the worker regenerates it on recovery.
            state = plasma_runtime.initial_state(config)
            planned, insertion, target = plasma_runtime.plan(config, state)
            (task / "input.lammps").write_text(
                plasma_runtime.block_input(
                    config, planned, None, "checkpoints/preview", insertion, target, 0
                )
            )


def _spread(groups, budget, maximum, separation):
    selected = []
    accepted = defaultdict(list)
    queues = {}
    for key, frames in sorted(groups.items()):
        queues[key] = sorted(frames, key=lambda item: (-item[2], item[1]["time"]))
    while len(selected) < budget:
        progress = False
        for key, queue in queues.items():
            if len(accepted[key]) >= maximum:
                continue
            while queue:
                item = queue.pop(0)
                if all(
                    abs(item[1]["time"] - other) >= separation
                    for other in accepted[key]
                ):
                    selected.append(item)
                    accepted[key].append(item[1]["time"])
                    progress = True
                    break
            if len(selected) >= budget:
                break
        if not progress:
            break
    return selected


def select_frames(tasks, job, jdata, system, total_budget):
    """Return ordinary FP frame references plus an explicit, disjoint holdout map."""
    options, budgets = selection_budgets(job, total_budget)
    f_lo = limit(
        job.get("model_devi_f_trust_lo", jdata["model_devi_f_trust_lo"]), system
    )
    f_hi = limit(
        job.get("model_devi_f_trust_hi", jdata["model_devi_f_trust_hi"]), system
    )
    v_lo = limit(
        job.get("model_devi_v_trust_lo", jdata.get("model_devi_v_trust_lo", 1e10)),
        system,
    )
    v_hi = limit(
        job.get("model_devi_v_trust_hi", jdata.get("model_devi_v_trust_hi", 1e10)),
        system,
    )
    candidate, accurate, failed = [], [], []
    groups = defaultdict(list)
    eligible = defaultdict(list)
    metadata = {}
    for task in tasks:
        frames = json.loads((Path(task) / "frames.json").read_text())
        for row in np.atleast_2d(np.loadtxt(Path(task) / "model_devi.out")):
            if row.shape[0] < 7 or row[0] < jdata.get("model_devi_skip", 0):
                continue
            step = int(row[0])
            ref = [str(task), step]
            frame = frames.get(str(step))
            if not frame:
                raise ValueError(f"Missing plasma frame metadata: {task} step {step}")
            key = (str(task), step)
            metadata[key] = dict(
                frame,
                validation=False,
                selected_for_fp=False,
                source_task=str(task),
                max_devi_f=float(row[4]) if np.isfinite(row[4]) else None,
                max_devi_v=float(row[1]) if np.isfinite(row[1]) else None,
            )
            if not np.isfinite(row[:7]).all() or row[4] >= f_hi or row[1] >= v_hi:
                metadata[key]["classification"] = "failed"
                failed.append(ref)
                continue
            kind = frame["kind"]
            bucket = (kind, str(task), frame["event"])
            item = (ref, frame, float(row[4]))
            eligible[bucket].append(item)
            if row[4] >= f_lo or row[1] >= v_lo:
                metadata[key]["classification"] = "candidate"
                candidate.append(ref)
                groups[bucket].append(item)
            else:
                metadata[key]["classification"] = "accurate"
                accurate.append(ref)
    # Independent event-diverse labels are selected before training candidates.
    # Seeded priorities do not depend on committee disagreement.
    rng = random.Random(repr(job.get("plasma", {}).get("seed", 42)))
    validation_groups = {}
    exposure_available = any(key[0] != "eq" for key in eligible)
    # Keep a candidate event from each enabled phase available for learning;
    # in a one-ion segment, the sole impact must not all become validation.
    reserved_events = set()
    for kind in ("eq", "ion", "neutral"):
        enabled = (
            budgets[kind] > 0 if options["_legacy"] else options[f"{kind}_fraction"] > 0
        )
        matching = sorted(key for key in groups if key[0] == kind)
        if enabled and matching:
            reserved_events.add(matching[0])
    for key, frames in eligible.items():
        if key in reserved_events or (exposure_available and key[0] == "eq"):
            continue
        validation_groups[key] = [
            (ref, frame, rng.random()) for ref, frame, _ in frames
        ]
    heldout = _spread(validation_groups, budgets["validation"], 1, 0)

    def event_key(item):
        return (item[1]["kind"], item[0][0], item[1]["event"])

    heldout_events = {event_key(item) for item in heldout}
    phase_selection = []
    for kind in ("eq", "ion", "neutral"):
        phase_groups = {
            key: list(values)
            for key, values in groups.items()
            if key[0] == kind
            and key not in heldout_events
            and (options["_legacy"] or options[f"{kind}_fraction"] > 0)
        }
        phase_selection.append(
            _spread(
                phase_groups,
                budgets[kind] if options["_legacy"] else total_budget,
                options["max_frames_per_event"],
                options["min_time_separation"],
            )
        )
    selected = []
    if options["_legacy"]:
        for values in itertools.zip_longest(*phase_selection):
            for item in values:
                if item is not None and len(selected) < total_budget:
                    selected.append(item)
    else:
        kinds = ("eq", "ion", "neutral")
        used = {}
        for kind, values in zip(kinds, phase_selection):
            count = min(budgets[kind], len(values))
            selected.extend(values[:count])
            used[kind] = count
        # Unavailable phase/holdout slots go to other enabled phases. Prefer
        # exposure when borrowing; never defeat event/time-separation limits.
        training_budget = total_budget - len(heldout)
        pools = dict(zip(kinds, phase_selection))
        while len(selected) < training_budget:
            progress = False
            for kind in ("ion", "neutral", "eq"):
                if options[f"{kind}_fraction"] > 0 and used[kind] < len(pools[kind]):
                    selected.append(pools[kind][used[kind]])
                    used[kind] += 1
                    progress = True
                    if len(selected) == training_budget:
                        break
            if not progress:
                break
    for item in heldout:
        metadata[(item[0][0], item[0][1])]["validation"] = True
    selected = [item[0] for item in selected + heldout]
    for task, step in selected:
        metadata[(task, step)]["selected_for_fp"] = True
    counter = Counter(
        accurate=len(accurate), candidate=len(candidate), failed=len(failed)
    )
    return accurate, selected, failed, counter, metadata


def run_tasks(iter_index, jdata, mdata):
    import shlex

    from dpgen.dispatcher.Dispatcher import make_submission
    from dpgen.generator.lib.utils import check_api_version
    from dpgen.generator.run import _get_model_suffix

    check_api_version(mdata)
    root = Path(make_iter_name(iter_index)) / "01.model_devi"
    models = sorted(
        path.name for path in root.glob(f"graph*{_get_model_suffix(jdata)}")
    )
    command = "python plasma_runtime.py --command " + shlex.quote(
        mdata["model_devi_command"]
    )
    submission = make_submission(
        mdata["model_devi_machine"],
        mdata["model_devi_resources"],
        commands=[command],
        work_path=str(root),
        run_tasks=sorted(path.name for path in root.glob("task.*.*")),
        group_size=mdata["model_devi_group_size"],
        forward_common_files=models,
        forward_files=[
            "conf.lmp",
            "plasma.json",
            "plasma_runtime.py",
            "checkpoints",
            "traj",
        ],
        backward_files=[
            "model_devi.out",
            "frames.json",
            "deposition.json",
            "plasma.status.json",
            "traj",
            "checkpoints",
        ],
        outlog="model_devi.log",
        errlog="model_devi.log",
    )
    submission.run_submission()


def link_potcars(iter_index, jdata):
    """Choose pseudopotentials per snapshot, including Cu-only equilibration."""
    root = Path(make_iter_name(iter_index)) / "02.fp"
    pseudopotentials = Path(jdata["fp_pp_path"]).resolve()
    files = dict(zip(jdata["type_map"], jdata["fp_pp_files"]))
    if len(files) != len(jdata["type_map"]):
        raise ValueError("fp_pp_files must contain one pseudopotential per element")
    for task in sorted(root.glob("task.*.*")):
        structure = dpdata.System(str(task / "POSCAR"), fmt="vasp/poscar")
        elements = structure.data["atom_names"]
        system = task.name.split(".")[1]
        combined = root / f"POTCAR.{system}.{'-'.join(elements)}"
        if not combined.exists():
            with combined.open("wb") as output:
                for element in elements:
                    with (pseudopotentials / files[element]).open("rb") as source:
                        shutil.copyfileobj(source, output)
        (task / "POTCAR").symlink_to(Path("..") / combined.name)


def log_fp_origin(fp_task, frame_job):
    """Log one created FP task and return its durable provenance record."""
    frame = frame_job["plasma_frame"]
    origin = {
        "fp_task": str(fp_task),
        "source_task": frame["source_task"],
        "step": frame["step"],
        "time_ps": frame["time"],
        "phase": frame["phase"],
        "kind": frame["kind"],
        "event": frame["event"],
        "since_arrival_ps": frame.get("since_arrival"),
        "temperature_K": frame_job["temps"],
        "surface": frame_job["plasma"]["surface"],
        "incident_energy_eV": frame_job["plasma"]["eion"]
        if frame["kind"] == "ion"
        else None,
        "assignment": "validation" if frame["validation"] else "training",
        "classification": frame["classification"],
        "max_devi_f": frame["max_devi_f"],
        "max_devi_v": frame["max_devi_v"],
    }
    dlog.info(
        "Plasma FP %s <- %s step=%d time=%.9g ps kind=%s event=%s "
        "temperature=%s K surface=%s %s classification=%s max_devi_f=%s max_devi_v=%s",
        origin["fp_task"],
        origin["source_task"],
        origin["step"],
        origin["time_ps"],
        origin["kind"],
        origin["event"],
        origin["temperature_K"],
        origin["surface"],
        origin["assignment"],
        origin["classification"],
        origin["max_devi_f"],
        origin["max_devi_v"],
    )
    return origin


def _selection_counts(frames, origins):
    frames = list(frames)
    origins = list(origins)
    counts = {
        key: sum(frame["classification"] == key for frame in frames)
        for key in ("accurate", "candidate", "failed")
    }
    counts["sampled"] = len(frames)
    counts["candidate_rate"] = counts["candidate"] / len(frames) if frames else 0.0
    counts["failed_rate"] = counts["failed"] / len(frames) if frames else 0.0
    for assignment in ("training", "validation"):
        counts[f"{assignment}_selected"] = sum(
            frame["selected_for_fp"]
            and frame["validation"] == (assignment == "validation")
            for frame in frames
        )
        counts[f"{assignment}_created"] = sum(
            origin["assignment"] == assignment for origin in origins
        )
    counts["fp_created"] = len(origins)
    for name in ("max_devi_f", "max_devi_v"):
        finite = [frame[name] for frame in frames if frame[name] is not None]
        counts[name] = max(finite) if finite else None
    return counts


def write_selection_report(output, system, job, budget, metadata, origins):
    """Report observed uncertainty separately from actual labeling allocation."""
    output = Path(output)
    options, requested = selection_budgets(job, budget)
    frames = list(metadata.values())
    total = _selection_counts(frames, origins)
    report = {
        "system": system,
        "budget_mode": "absolute" if options["_legacy"] else "fractional",
        "fp_task_max": budget,
        "requested_budgets": requested,
        "fractions": None
        if options["_legacy"]
        else {
            kind: options[f"{kind}_fraction"]
            for kind in ("eq", "ion", "neutral", "validation")
        },
        "totals": total,
        "phases": {},
        "events": [],
        "origins": origins,
        "scope": "Global committee deviations grouped by exposure phase/event; not spatial atom-wise uncertainty.",
    }
    for kind in ("eq", "ion", "neutral"):
        counts = _selection_counts(
            (frame for frame in frames if frame["kind"] == kind),
            (origin for origin in origins if origin["kind"] == kind),
        )
        counts["fraction_of_candidates"] = (
            counts["candidate"] / total["candidate"] if total["candidate"] else 0.0
        )
        counts["fraction_of_fp"] = (
            counts["fp_created"] / total["fp_created"] if total["fp_created"] else 0.0
        )
        counts["fraction_of_budget"] = counts["fp_created"] / budget if budget else 0.0
        report["phases"][kind] = counts
        dlog.info(
            "Plasma selection system %03d %s: sampled=%d accurate=%d candidate=%d "
            "(%.1f%%) failed=%d (%.1f%%) FP training=%d validation=%d share=%.1f%%",
            system,
            kind,
            counts["sampled"],
            counts["accurate"],
            counts["candidate"],
            100 * counts["candidate_rate"],
            counts["failed"],
            100 * counts["failed_rate"],
            counts["training_created"],
            counts["validation_created"],
            100 * counts["fraction_of_fp"],
        )
    groups = defaultdict(list)
    for frame in frames:
        groups[(frame["source_task"], frame["kind"], frame["event"])].append(frame)
    for (source, kind, event), values in sorted(groups.items()):
        counts = _selection_counts(
            values,
            (
                origin
                for origin in origins
                if origin["source_task"] == source
                and origin["event"] == event
                and origin["kind"] == kind
            ),
        )
        report["events"].append(
            dict(
                source_task=source,
                kind=kind,
                event=event,
                time_start_ps=min(frame["time"] for frame in values),
                time_end_ps=max(frame["time"] for frame in values),
                **counts,
            )
        )
    text = [
        f"Plasma selection: system {system:03d}",
        f"Budget mode: {report['budget_mode']}; fp_task_max={budget}",
        "Requested: "
        + ", ".join(f"{kind}={value}" for kind, value in requested.items()),
        "Candidate/failure rates are fractions of sampled frames; phase sampling frequencies can differ.",
        "This report groups exposure stages/events, not spatial regions of the slab.",
        "",
        "phase sampled accurate candidate candidate_% failed failed_% training_FP validation_FP FP_share_%",
    ]
    for kind, counts in report["phases"].items():
        text.append(
            f"{kind} {counts['sampled']} {counts['accurate']} {counts['candidate']} {100 * counts['candidate_rate']:.2f} "
            f"{counts['failed']} {100 * counts['failed_rate']:.2f} {counts['training_created']} "
            f"{counts['validation_created']} {100 * counts['fraction_of_fp']:.2f}"
        )
    text.extend(
        [
            "",
            "event source kind sampled accurate candidate failed training_FP validation_FP",
        ]
    )
    for event in report["events"]:
        text.append(
            f"{event['event']} {event['source_task']} {event['kind']} {event['sampled']} "
            f"{event['accurate']} {event['candidate']} {event['failed']} "
            f"{event['training_created']} {event['validation_created']}"
        )
    text.extend(["", "FP origins:"])
    for origin in origins:
        text.append(
            f"{origin['fp_task']} <- {origin['source_task']} step={origin['step']} time={origin['time_ps']:.9g} ps "
            f"{origin['kind']} event={origin['event']} {origin['assignment']} {origin['classification']}"
        )
    (output / f"plasma-selection.{system:03d}.log").write_text("\n".join(text) + "\n")
    plasma_runtime.atomic_json(output / f"plasma-selection.{system:03d}.json", report)
    columns = (
        list(origins[0])
        if origins
        else ["fp_task", "source_task", "step", "kind", "event", "assignment"]
    )
    with (output / f"plasma-origins.{system:03d}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(origins)
    dlog.info(
        "Plasma selection system %03d budget=%d FP training=%d validation=%d; report=%s",
        system,
        budget,
        total["training_created"],
        total["validation_created"],
        output / f"plasma-selection.{system:03d}.log",
    )
    return report
