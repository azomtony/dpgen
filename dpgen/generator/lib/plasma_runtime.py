"""Standalone, restartable plasma worker (standard library only).

Copied into each task so batch nodes need Python and the configured LAMMPS
executable, but do not need a separate DP-GEN or LAMMPS Python installation.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shlex
import subprocess
from pathlib import Path

FIELDS = "id type x y z vx vy vz fx fy fz"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def seed_for(seed, *parts):
    value = hashlib.sha256(repr((seed, parts)).encode()).hexdigest()
    return int(value[:16], 16) % 899999999 + 1


def initial_state(config):
    eq_elapsed = config.get("eq_elapsed", 0.0)
    remaining = max(0.0, config["eq_time"] - eq_elapsed)
    eq_complete = config.get("eq_complete", False) or remaining == 0
    eq_target = 0.0 if eq_complete else remaining
    return {
        "version": 1,
        "parent": None,
        "step": 0,
        "time": 0.0,
        "dt": config["initial_dt"],
        "phase": "exposure" if eq_complete else "eq",
        "eq_complete": eq_complete,
        "eq_target": eq_target,
        "exposure_start": eq_target,
        "final_time": eq_target + config["nions"] * config["ion_dt"],
        "arrivals": 0,
        "active_event": None,
        "status": "running",
        "eq_elapsed_initial": eq_elapsed,
    }


def plan(config, state):
    """Plan one transactional block without changing published progress."""
    new = copy.deepcopy(state)
    insertion = None
    if new["phase"] == "eq":
        target = new["eq_target"]
    else:
        if new["active_event"] is None:
            if new["arrivals"] == config["nions"] * (1 + config["ratio"]):
                new["status"] = "complete"
                return new, None, new["time"]
            event = new["arrivals"] + 1
            new["active_event"] = event
            new["arrivals"] = event
            rng = random.Random(seed_for(config["seed"], "arrival", event))
            is_ion = (event - 1) % (config["ratio"] + 1) == 0
            sigma = math.sqrt(
                8.617333262e-5 * config["temp"] / (15.999 * 1.036426965e-4)
            )
            velocity = [
                rng.gauss(0, sigma),
                rng.gauss(0, sigma),
                -abs(rng.gauss(0, sigma)),
            ]
            if is_ion:
                velocity = [
                    0.0,
                    0.0,
                    -math.sqrt(2 * config["eion"] / (15.999 * 1.036426965e-4)),
                ]
            insertion = {
                "event": event,
                "kind": "ion" if is_ion else "neutral",
                "velocity": velocity,
            }
        target = new["exposure_start"] + new["active_event"] * config["ion_dt"] / (
            1 + config["ratio"]
        )
    return new, insertion, target


def advance(config, state, step, time, dt, stopped=False):
    if (
        not math.isfinite(time)
        or not math.isfinite(dt)
        or time <= state["time"]
        or dt <= 0
    ):
        raise RuntimeError("LAMMPS did not advance to a valid physical time")
    state.update(step=int(step), time=time, dt=dt)
    if stopped:
        state["status"] = "trust_stop"
    elif state["phase"] == "eq" and time >= state["eq_target"]:
        state.update(phase="exposure", eq_complete=True)
    elif state["phase"] == "exposure" and state["active_event"] is not None:
        target = state["exposure_start"] + state["active_event"] * config["ion_dt"] / (
            1 + config["ratio"]
        )
        if time >= target:
            state["active_event"] = None
            if state["arrivals"] == config["nions"] * (1 + config["ratio"]):
                state["status"] = "complete"
    return state


def frame_info(config, state, step, time, phase):
    event = state["active_event"] if phase == "exposure" else 0
    return {
        "step": step,
        "time": time,
        "phase": phase,
        "event": event,
        "kind": ("ion" if (event - 1) % (config["ratio"] + 1) == 0 else "neutral")
        if event
        else "eq",
        "since_arrival": max(0.0, time - state["arrival_time"]) if event else None,
        "eq_elapsed": min(config["eq_time"], state["eq_elapsed_initial"] + time)
        if phase == "eq"
        else config["eq_time"],
        "eq_complete": phase != "eq",
        "arrivals": state["arrivals"],
        "cumulative_arrivals": config.get("cumulative_arrivals", 0) + state["arrivals"],
        "cumulative_ions": config.get("cumulative_ions", 0)
        + (state["arrivals"] + config["ratio"]) // (1 + config["ratio"]),
        "cumulative_neutrals": config.get("cumulative_neutrals", 0)
        + state["arrivals"]
        - (state["arrivals"] + config["ratio"]) // (1 + config["ratio"]),
    }


def dump_time(path):
    lines = Path(path).read_text().splitlines()
    for index, line in enumerate(lines):
        if line == "ITEM: TIME":
            return float(lines[index + 1])
    raise ValueError(f"Missing physical time in {path}")


def read_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            row = [float(value) for value in line.split()]
            if len(row) < 7:
                raise ValueError(f"Invalid model-deviation record: {line}")
            rows.append(row)
    return rows


def over_stop(row, config):
    return (
        not all(math.isfinite(value) for value in row[:7])
        or row[4] >= config["f_hi"] * config["stop_factor"]
        or row[1] >= config["v_hi"] * config["stop_factor"]
    )


def block_input(config, state, previous, directory, insertion, target, serial):
    """Reconstruct the user's physical controls, retaining checkpoint velocities."""
    p = Path(directory).as_posix()
    frequency = config["eq_trj_freq"] if state["phase"] == "eq" else config["trj_freq"]
    seed = seed_for(config["seed"], "bath", state["step"], state["active_event"])
    read = (
        f'read_restart "{previous}/restart.bin"' if previous else 'read_data "conf.lmp"'
    )
    reset = "" if previous else "reset_timestep 0 time 0.0"
    model_list = " ".join(config["models"])
    thermostat_group = "mobile" if state["phase"] == "eq" else "bath"
    thermostat_temp = "TCu" if state["phase"] == "eq" else "Tbath"
    text = f"""clear
units metal
atom_style atomic
boundary p p f
atom_modify map yes
{read}
{reset}
pair_style deepmd {model_list} out_freq {frequency} out_file {p}/model_devi.out
pair_coeff * * Cu O
mass 1 63.546
mass 2 15.999
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
group copper type 1
group oxygen type 2
variable dz equal {config["a"]}*{ {100: 0.5, 110: 1 / (2 * math.sqrt(2)), 111: 1 / math.sqrt(3)}[config["surface"]] }
variable zcut equal bound(copper,zmin)+1.5*v_dz
if "${{zcut}} >= $(bound(copper,zmax))" then "quit 1"
region bottom block INF INF INF INF INF ${{zcut}} units box
group bottom_candidates region bottom
group frozen intersect copper bottom_candidates
group mobile subtract copper frozen
group moving subtract all frozen
velocity frozen set 0.0 0.0 0.0
fix hold frozen setforce 0.0 0.0 0.0
fix integrate moving nve
variable bxlo equal xlo+{config["edge"]}
variable bxhi equal xhi-{config["edge"]}
variable bylo equal ylo+{config["edge"]}
variable byhi equal yhi-{config["edge"]}
variable materialtop equal bound(copper,zmax)+2.0
region left block INF ${{bxlo}} INF INF INF v_materialtop units box
region right block ${{bxhi}} INF INF INF INF v_materialtop units box
region front block INF INF INF ${{bylo}} INF v_materialtop units box
region back block INF INF ${{byhi}} INF INF v_materialtop units box
region bath_region union 4 left right front back
group bath dynamic moving region bath_region every 1
compute TCu mobile temp
compute Tbath bath temp
compute_modify Tbath dynamic/dof yes
compute_modify thermo_temp dynamic/dof yes
fix heat {thermostat_group} langevin {config["temp"]} {config["temp"]} 0.1 {seed} zero yes
fix_modify heat temp {thermostat_temp}
variable zescape equal zhi-5.0
variable zbottom equal zlo+1.0
if "$(bound(copper,zmin)) <= ${{zbottom}}" then "quit 1"
region escape_top block INF INF INF INF ${{zescape}} INF units box
region escape_bottom block INF INF INF INF INF ${{zbottom}} units box
region escape union 2 escape_top escape_bottom
fix removeO oxygen evaporate 1 1000000 escape {seed}
timestep {state["dt"]:.17g}
fix adaptive moving dt/reset 1 NULL {config["max_dt"]} {config["max_displacement"]} units box
variable clock equal time
thermo {frequency}
thermo_style custom step time dt atoms c_TCu c_Tbath pe ke etotal
thermo_modify lost error flush yes
dump trajectory all custom {frequency} {p}/traj/*.lammpstrj {FIELDS}
dump_modify trajectory sort id time yes
"""
    if insertion:
        vx, vy, vz = insertion["velocity"]
        text += f"""variable ixlo equal xlo+{config["edge"] + config["margin"]}
variable ixhi equal xhi-{config["edge"] + config["margin"]}
variable iylo equal ylo+{config["edge"] + config["margin"]}
variable iyhi equal yhi-{config["edge"] + config["margin"]}
if "${{ixlo}} >= ${{ixhi}} || ${{iylo}} >= ${{iyhi}}" then "quit 1"
variable zinlo equal bound(copper,zmax)+{config["height"]}
variable zinhi equal v_zinlo+0.5
if "${{zinhi}} >= ${{zescape}}" then "quit 1"
region injection block ${{ixlo}} ${{ixhi}} ${{iylo}} ${{iyhi}} ${{zinlo}} ${{zinhi}} units box
fix addO moving deposit 1 2 1 {seed_for(config["seed"], "position", insertion["event"])} region injection near 2.0 attempt 1000 vx {vx} {vx} vy {vy} {vy} vz {vz} {vz} units box
timestep {config["initial_dt"]}
run 1 post no
if "$(f_addO) != 1" then "quit 1"
group oxygen type 2
unfix addO
region injection delete
print "$(time:%.17g)" file {p}/arrival.txt screen no
"""
    # Absolute time boundaries survive restarts; never run a fresh full interval.
    text += f"""fix block_end all halt 1 v_clock >= {target:.17g} error continue message no
variable checkpoint_end internal $(step+{config["checkpoint_freq"]}:%.0f)
variable plasma_abort internal 0
label plasma_sample
if "$(time) >= {target:.17g}" then "jump SELF plasma_block_done"
variable sample_steps equal min({frequency},v_checkpoint_end-step)
run ${{sample_steps}} post no
shell python plasma_runtime.py --monitor {p}
include {p}/control.lammps
if "${{plasma_abort}} > 0 || $(time) >= {target:.17g} || $(step) >= ${{checkpoint_end}}" then "jump SELF plasma_block_done"
jump SELF plasma_sample
label plasma_block_done
unfix block_end
write_restart {p}/restart.bin
print "$(step:%.0f) $(time:%.17g) $(dt:%.17g) $(count(oxygen):%.0f)" file {p}/progress.txt screen no
"""
    return text


def checkpoint_chain(latest):
    chain = []
    while latest:
        path = Path(latest)
        state = json.loads((path / "state.json").read_text())
        chain.append(path)
        latest = state["parent"]
    return list(reversed(chain))


def recover(config_hash):
    for path in sorted(Path("checkpoints").glob("[0-9]*"), reverse=True):
        try:
            state = json.loads((path / "state.json").read_text())
            if state["config_hash"] != config_hash:
                raise RuntimeError(
                    "Plasma settings changed for an existing task; use a new task directory"
                )
            if digest(path / "restart.bin") != state["restart_sha256"]:
                continue
            checkpoint_chain(path)  # Validate ancestry before publishing outputs.
            return path, state
        except (OSError, ValueError, KeyError):
            continue
    if any(Path("checkpoints").glob("*/state.json")):
        raise RuntimeError(
            "No complete valid plasma checkpoint remains; refusing to restart the dose from scratch"
        )
    return None, None


def publish(latest, incremental=False):
    """Publish a block, or rebuild committed outputs after an interrupted tail."""
    Path("traj").mkdir(exist_ok=True)
    if incremental and all(
        Path(name).exists()
        for name in ("model_devi.out", "frames.json", "deposition.json")
    ):
        rows = {int(row[0]): row for row in read_rows("model_devi.out")}
        frames = {
            int(key): value
            for key, value in json.loads(Path("frames.json").read_text()).items()
        }
        events = json.loads(Path("deposition.json").read_text())
        chain = [Path(latest)]
    else:
        rows, frames, events = {}, {}, []
        chain = checkpoint_chain(latest)
    for path in chain:
        block = json.loads((path / "samples.json").read_text())
        for frame in block["frames"]:
            step = frame["step"]
            rows[step] = frame.pop("deviation")
            frames[step] = frame
            source = path / "traj" / f"{step}.lammpstrj"
            destination = Path("traj") / source.name
            temporary = destination.with_suffix(".tmp")
            temporary.unlink(missing_ok=True)
            os.link(source, temporary)
            os.replace(temporary, destination)
        if block["insertion"]:
            events.append(block["insertion"])
    for path in Path("traj").glob("*.lammpstrj"):
        if int(path.stem) not in frames:
            path.unlink()
    temporary = Path("model_devi.out.tmp")
    with temporary.open("w") as stream:
        stream.write(
            "# step max_devi_v min_devi_v avg_devi_v max_devi_f min_devi_f avg_devi_f\n"
        )
        for step in sorted(rows):
            stream.write(" ".join(str(value) for value in rows[step]) + "\n")
    os.replace(temporary, "model_devi.out")
    atomic_json("frames.json", {str(key): value for key, value in frames.items()})
    atomic_json("deposition.json", events)


def run(command, config_file="plasma.json", execute=subprocess.run):
    config = json.loads(Path(config_file).read_text())
    config_hash = hashlib.sha256(
        (json.dumps(config, sort_keys=True) + digest("conf.lmp")).encode()
    ).hexdigest()
    Path("checkpoints").mkdir(exist_ok=True)
    previous, state = recover(config_hash)
    if state is None:
        state = initial_state(config)
    else:
        publish(previous)
    serial = max([int(p.name) for p in Path("checkpoints").glob("[0-9]*")] + [0])
    while state["status"] == "running":
        planned, insertion, target = plan(config, state)
        if planned["status"] == "complete":
            state = planned
            break
        serial += 1
        directory = Path("checkpoints") / f"{serial:09d}"
        directory.mkdir()
        (directory / "traj").mkdir()
        input_path = directory / "input.lammps"
        input_path.write_text(
            block_input(config, planned, previous, directory, insertion, target, serial)
        )
        # A nonzero exit leaves this uncommitted block for diagnosis. The next
        # invocation recovers the last published checkpoint, not these files.
        execute(shlex.split(command) + ["-i", str(input_path)], check=True)
        progress = (directory / "progress.txt").read_text().split()
        step, time, dt = int(progress[0]), float(progress[1]), float(progress[2])
        if insertion:
            planned["arrival_time"] = float((directory / "arrival.txt").read_text())
        stopped = False
        samples = []
        for row in read_rows(directory / "model_devi.out"):
            frame_step = int(row[0])
            if frame_step <= state["step"]:
                continue
            dump = directory / "traj" / f"{frame_step}.lammpstrj"
            if not dump.exists():
                raise RuntimeError(
                    f"Committee record has no matching trajectory: {dump}"
                )
            info = frame_info(
                config, planned, frame_step, dump_time(dump), planned["phase"]
            )
            info["deviation"] = [
                value if math.isfinite(value) else "nan" for value in row
            ]
            samples.append(info)
            if over_stop(row, config):
                stopped = True
                planned["stop_frame"] = frame_step
                break
        if not samples:
            # Short event-boundary blocks may contain no periodic sample. They
            # still checkpoint progress; sampling resumes on the same step grid.
            pass
        if insertion:
            insertion.update(
                time=planned["arrival_time"],
                seed=config["seed"],
                incident_eV=0.5
                * 15.999
                * 1.036426965e-4
                * sum(v * v for v in insertion["velocity"]),
            )
        committed = advance(config, planned, step, time, dt, stopped)
        committed["retained_oxygen"] = int(progress[3])
        committed.update(
            parent=str(previous) if previous else None,
            config_hash=config_hash,
            restart_sha256=digest(directory / "restart.bin"),
        )
        atomic_json(
            directory / "samples.json", {"frames": samples, "insertion": insertion}
        )
        atomic_json(directory / "state.json", committed)  # Commit marker, written last.
        previous, state = directory, committed
        publish(previous, incremental=True)
        # Keep the latest two binary checkpoints. Frame hardlinks and small
        # manifests remain for labeling and recovery of the output history.
        chain = checkpoint_chain(previous)
        for old in chain[:-2]:
            (old / "restart.bin").unlink(missing_ok=True)
    atomic_json("plasma.status.json", state)
    return state


def monitor(directory, config_file="plasma.json"):
    """Check committee output between short runs without reloading models."""
    control = Path(directory) / "control.lammps"
    # Fail closed if parsing or the monitor subprocess fails.
    control.write_text("variable plasma_abort internal 1\n")
    config = json.loads(Path(config_file).read_text())
    output = Path(directory) / "model_devi.out"
    stop = output.exists() and any(over_stop(row, config) for row in read_rows(output))
    control = Path(directory) / "control.lammps"
    temporary = control.with_suffix(".tmp")
    temporary.write_text(f"variable plasma_abort internal {int(stop)}\n")
    os.replace(temporary, control)
    return stop


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--command")
    group.add_argument("--monitor")
    args = parser.parse_args()
    if args.monitor:
        monitor(args.monitor)
    else:
        run(args.command)


if __name__ == "__main__":
    main()
