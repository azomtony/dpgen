"""Continue native LAMMPS exploration from a trusted previous frame."""

import json
import os
import tempfile
from pathlib import Path

import dpdata
import numpy as np

from dpgen import dlog


def _key(sys_idx, source, job):
    return (
        sys_idx,
        os.path.realpath(source),
        job.get("ensemble"),
        job.get("temps"),
        job.get("press"),
        job.get("ele_temp"),
        job.get("pka_e"),
    )


def previous_tasks(iter_index):
    """Index the immediate previous iteration, including legacy task metadata."""
    tasks = {}
    if iter_index == 0:
        return tasks
    root = Path(f"iter.{iter_index - 1:06d}") / "01.model_devi"
    for task in sorted(root.glob("task.*.*")):
        try:
            job = json.loads((task / "job.json").read_text())
            source = job.get("source_conf")
            if source is None:
                # Older native tasks link conf.lmp to ../confs/SSS.CCC.lmp.
                conf = task / "conf.lmp"
                if not conf.is_symlink():
                    continue
                poscar = (task / os.readlink(conf)).with_suffix(".poscar")
                if not poscar.exists():
                    continue
                source = str(poscar.resolve())
            key = _key(int(task.name.split(".")[1]), source, job)
            tasks.setdefault(key, []).append(task)
        except (OSError, ValueError, TypeError) as exc:
            dlog.warning("Cannot match exploration restart task %s: %s", task, exc)
    return tasks


def _limit(jdata, job, name, sys_idx, default=None):
    value = job.get(name)
    if value is None:
        value = jdata.get(name, default)
    if isinstance(value, dict):
        value = value[str(sys_idx)]
    elif isinstance(value, list):
        value = value[sys_idx]
    return float(value)


def _dump_offsets(path):
    """Index real timesteps rather than assuming a uniform dump frequency."""
    offsets = {}
    with path.open() as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if line.strip() == "ITEM: TIMESTEP":
                try:
                    offsets[int(stream.readline())] = offset
                except ValueError:
                    continue
    return offsets


def _read_frame(path, type_map, offset=None):
    if offset is None:
        return dpdata.System(str(path), fmt="lammps/dump", type_map=type_map)
    # Extract a single merged frame so a truncated final frame cannot prevent
    # reading an earlier, usable frame. Keep memory bounded by one structure.
    with path.open() as stream, tempfile.NamedTemporaryFile(mode="w+") as frame:
        stream.seek(offset)
        frame.write(stream.readline())
        for line in stream:
            if line.strip() == "ITEM: TIMESTEP":
                break
            frame.write(line)
        frame.flush()
        return dpdata.System(frame.name, fmt="lammps/dump", type_map=type_map)


def restart_from_previous(index, iter_index, sys_idx, source, job, jdata, destination):
    """Write the latest usable matching frame; return provenance or fallback reason."""
    matches = index.get(_key(sys_idx, source, job), [])
    fallback = {"source": "initial", "reason": "no matching previous task"}
    if not matches:
        return fallback

    previous_job = jdata["model_devi_jobs"][iter_index - 1]
    f_hi = _limit(jdata, previous_job, "model_devi_f_trust_hi", sys_idx)
    v_hi = _limit(jdata, previous_job, "model_devi_v_trust_hi", sys_idx, 1e10)
    skip = jdata.get("model_devi_skip", 0)
    candidates = []
    for task in matches:
        try:
            normalized = task / "model_devi_avgf.out"
            if jdata.get("model_devi_f_avg_relative", False):
                if normalized.exists():
                    deviations = np.loadtxt(normalized)
                else:
                    from dpgen.generator.run import _read_model_devi_file

                    deviations = _read_model_devi_file(
                        str(task), True, jdata.get("model_devi_merge_traj", False)
                    )
            else:
                deviations = np.loadtxt(task / "model_devi.out")
            deviations = np.atleast_2d(deviations)
            if deviations.shape[1] < 7:
                raise ValueError("expected at least seven model-deviation columns")
            merged = task / "all.lammpstrj"
            offsets = _dump_offsets(merged) if merged.exists() else {}
            for row in deviations:
                if (
                    np.isfinite(row[[0, 1, 4]]).all()
                    and row[0] >= skip
                    and row[0] == int(row[0])
                    and 0 <= row[4] < f_hi
                    and 0 <= row[1] < v_hi
                ):
                    step = int(row[0])
                    dump = task / "traj" / f"{step}.lammpstrj"
                    if dump.exists():
                        candidates.append((step, task, dump, None))
                    elif step in offsets:
                        candidates.append((step, task, merged, offsets[step]))
        except (OSError, ValueError, RuntimeError, AssertionError, IndexError) as exc:
            dlog.warning("Cannot read exploration restart data in %s: %s", task, exc)

    for step, task, dump, offset in sorted(
        candidates, key=lambda item: (item[0], str(item[1])), reverse=True
    ):
        try:
            system = _read_frame(dump, jdata["type_map"], offset)
            if (
                len(system) != 1
                or system.get_natoms() == 0
                or not np.isfinite(system.data["coords"]).all()
                or not np.isfinite(system.data["cells"]).all()
            ):
                raise ValueError("invalid restart structure")
            if jdata.get("model_devi_nopbc", False):
                system.remove_pbc()
            destination = Path(destination)
            with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
                output = Path(scratch) / "conf.lmp"
                system.to_lammps_lmp(str(output))
                os.replace(output, destination)
            dlog.info(
                "Exploration restart %s from %s frame %d", destination, task, step
            )
            return {"source": str(task), "step": step}
        except (
            OSError,
            ValueError,
            RuntimeError,
            AssertionError,
            IndexError,
            KeyError,
        ) as exc:
            dlog.warning(
                "Skipping unreadable restart frame %s step %d: %s", task, step, exc
            )
    fallback["reason"] = "no usable accurate or candidate frame in matching tasks"
    return fallback


def trajectory_cleanup_index(iter_index, clean_traj, restart):
    """Retain at least the latest iteration when continuation is enabled."""
    if clean_traj is False:
        return None
    if isinstance(clean_traj, bool):
        keep = 1 if restart else 0
    else:
        keep = max(clean_traj, 1) if restart else clean_traj
    index = iter_index - keep
    return index if index >= 0 else None
