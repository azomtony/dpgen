"""A pool of long GPU allocations that runs jobs side by side, one per GPU set.

Instead of one Slurm job per task, dpgen keeps a few long allocations (for
example 1 node x 4 GPUs x 24 h) and hands work to an allocation once it is
running. Inside each allocation ``fp_pilot.py`` starts the work on free GPUs
with ``CUDA_VISIBLE_DEVICES`` limited to that work's GPUs.

The pool can belong to one FP stage (``iter.*/02.fp/allocations``) or to the
whole run (``<workdir>/allocations``). A run-wide pool is shared by training,
exploration and single points: they never run at the same time within an
iteration, so the same nodes serve each stage in turn and are not given back
to the queue between stages.

Files under the pool directory:

``pool.json``          allocations, the GPUs each one has promised, and queued
                       dispatcher jobs (written only by the running dpgen)
``pool-config.json``   settings copied from machine.json ``pool``
``pool-events.log``    one line per allocation request, start, end, drain
``A01/``               one allocation: ``alloc.sbatch``, ``fp_pilot.py``,
                       ``status.json`` and the inbox/running/finished/cancel
                       folders read and written by the pilot
"""

import json
import math
import os
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

from dpgen import dlog, progress
from dpgen.dispatcher import fp_pilot

POOL_DIR = "allocations"
STATE = "pool.json"
CONFIG_FILE = "pool-config.json"
EVENTS = "pool-events.log"
ACTIVE = ("submitting", "queued", "running")
HISTORY_KEEP = 20

POOL_DEFAULTS = {
    "gpus": 4,
    "time": "24:00:00",
    "max_allocations": 1,
    "directives": [],
    "prepend": [],
    "vars": {},
    "python": None,
    "idle_exit": 1800,
    "start_margin": 600,
    "heartbeat_timeout": 600,
    "warn_before_end": 300,
    "safety_factor": 1.3,
    "runtime": {},
    "job_name": "dpgen-pool-{allocation}",
    "keep": 1,
    "renew_before": 0,
    "accounting_grace": 180,
}


def seconds(value):
    """Slurm-style duration (or a number of seconds) in seconds."""
    if isinstance(value, bool):
        raise ValueError(f"Not a duration: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.fullmatch(r"(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?", text)
    if not match:
        raise ValueError(f"Not a duration: {value!r}")
    days, first, second, third = match.groups()
    parts = [int(x) for x in (first, second, third) if x is not None]
    if days is not None:
        hours_, minutes, secs = (parts + [0, 0])[:3]
    elif len(parts) == 3:
        hours_, minutes, secs = parts
    elif len(parts) == 2:
        hours_, minutes, secs = 0, parts[0], parts[1]
    else:
        hours_, minutes, secs = 0, parts[0], 0
    return float(int(days or 0) * 86400 + hours_ * 3600 + minutes * 60 + secs)


def hours(value):
    return f"{value / 3600:.1f} h"


def _placeholders(template):
    from dpgen.dispatcher.fp_slurm import _placeholders as found

    return found(template)


def _positive_int(value):
    return not isinstance(value, bool) and isinstance(value, int) and value >= 1


def pool_options(value, label="pool"):
    """Validate machine.json ``pool`` and add ``*_seconds`` fields."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = sorted(set(value) - set(POOL_DEFAULTS))
    if unknown:
        raise ValueError(f"Invalid {label} options: {unknown}")
    config = dict(POOL_DEFAULTS, **value)
    config["vars"] = dict(config["vars"])
    if not _positive_int(config["gpus"]):
        raise ValueError(f"{label}.gpus must be a positive integer")
    limit = config["max_allocations"]
    if isinstance(limit, dict):
        if not limit or not all(_positive_int(v) for v in limit.values()):
            raise ValueError(
                f"{label}.max_allocations must map stage names to positive integers"
            )
    elif not _positive_int(limit):
        raise ValueError(f"{label}.max_allocations must be a positive integer")
    try:
        config["time_seconds"] = seconds(config["time"])
    except ValueError as exc:
        raise ValueError(f"{label}.time: {exc}") from exc
    if not isinstance(config["runtime"], dict):
        raise ValueError(f"{label}.runtime must map stage names to durations")
    try:
        config["runtime_seconds"] = {
            kind: seconds(text) for kind, text in config["runtime"].items()
        }
    except ValueError as exc:
        raise ValueError(f"{label}.runtime: {exc}") from exc
    for key in (
        "idle_exit",
        "start_margin",
        "heartbeat_timeout",
        "warn_before_end",
        "safety_factor",
        "accounting_grace",
    ):
        number = config[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number < 0
        ):
            raise ValueError(f"{label}.{key} must be a non-negative number")
    if config["safety_factor"] < 1:
        raise ValueError(f"{label}.safety_factor must be at least 1")
    if config["start_margin"] >= config["time_seconds"]:
        raise ValueError(f"{label}.start_margin must be shorter than {label}.time")
    keep = config["keep"]
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 0:
        raise ValueError(f"{label}.keep must be a non-negative integer")
    try:
        config["renew_before_seconds"] = seconds(config["renew_before"])
    except ValueError as exc:
        raise ValueError(f"{label}.renew_before: {exc}") from exc
    if config["renew_before_seconds"] >= config["time_seconds"]:
        raise ValueError(f"{label}.renew_before must be shorter than {label}.time")
    for key in ("directives", "prepend"):
        if not isinstance(config[key], list) or not all(
            isinstance(item, str) for item in config[key]
        ):
            raise ValueError(f"{label}.{key} must be a list of strings")
    for directive in config["directives"]:
        if not directive.startswith("-"):
            raise ValueError(
                f"{label}.directives must be sbatch options: {directive!r}"
            )
        if re.match(r"--(?:chdir|output|job-name|signal)\b|-[DoJ]\b", directive):
            raise ValueError(f"{directive!r} is set by dpgen; remove it from {label}")
    names = set(config["vars"]) | {"gpus", "time", "allocation"}
    for template in config["directives"] + [config["job_name"]]:
        missing = _placeholders(template) - names
        if missing:
            raise ValueError(
                f"{label} uses undefined variable(s) {sorted(missing)} in {template!r}"
            )
    if config["python"] is not None and not isinstance(config["python"], str):
        raise ValueError(f"{label}.python must be a path")
    return config


def pool_root(workdir):
    return Path(workdir).resolve() / POOL_DIR


def configure(workdir, mdata):
    """Copy machine.json ``pool`` next to the run-wide pool, for the Pilot machine.

    Returns the validated settings, or None when machine.json has no pool.
    """
    value = mdata.get("pool")
    if value is None:
        return None
    config = pool_options(value)
    root = pool_root(workdir)
    root.mkdir(parents=True, exist_ok=True)
    fp_pilot.atomic_json(root / CONFIG_FILE, value)
    return config


def load_config(root):
    path = Path(root) / CONFIG_FILE
    try:
        value = json.loads(path.read_text())
    except OSError as exc:
        raise RuntimeError(
            f'{path} not found. Add a top-level "pool" section to machine.json '
            "(see examples/run/plasma/README.md) and start dpgen again."
        ) from exc
    return pool_options(value)


def read_state(root):
    try:
        return json.loads((Path(root) / STATE).read_text())
    except (OSError, ValueError):
        return None


class NodePool:
    """Allocation lifecycle, GPU bookkeeping and generic job placement."""

    def __init__(
        self,
        root,
        config,
        slurm,
        clock=time.time,
        event=None,
        dry_run=False,
        workdir=None,
    ):
        self.root = Path(root)
        # dpgen run directory, for dpgen-live.log / dpgen-summary.log
        self.workdir = Path(workdir) if workdir else self.root.parent
        self.config = config
        self.slurm = slurm
        self.clock = clock
        self.dry_run = dry_run
        self.scripts = {}  # allocation id -> alloc.sbatch text (dry run)
        self._event = event
        self.state = read_state(self.root) or {
            "version": 1,
            "allocations": {},
            "items": {},
            "history": {},
        }
        for key in ("allocations", "items", "history"):
            self.state.setdefault(key, {})

    # ---- persistence -----------------------------------------------------

    @property
    def allocs(self):
        return self.state["allocations"]

    @property
    def items(self):
        return self.state["items"]

    def save(self):
        if self.dry_run:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.state["updated"] = self.clock()
        fp_pilot.atomic_json(self.root / STATE, self.state)

    def event(self, message, level="info"):
        if self._event is not None:
            self._event(message, level)
            return
        if self.dry_run:
            return
        getattr(dlog, level)("pool: %s", message)
        self.root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.clock()))
        with (self.root / EVENTS).open("a") as stream:
            stream.write(f"{stamp} {message}\n")

    # ---- views -----------------------------------------------------------

    def limit(self, kind):
        value = self.config["max_allocations"]
        if isinstance(value, dict):
            return value.get(kind, value.get("default", 1))
        return value

    def active(self):
        return {k: a for k, a in self.allocs.items() if a["status"] in ACTIVE}

    def used_gpus(self, alloc):
        return sum(item["ngpus"] for item in alloc.get("assigned", {}).values())

    def free_gpus(self, alloc):
        return alloc["gpus"] - self.used_gpus(alloc)

    def remaining(self, alloc):
        """Seconds left; a node not started or not yet checked in has its full time."""
        if alloc.get("deadline"):
            return alloc["deadline"] - self.clock()
        return alloc.get("time_limit", self.config["time_seconds"])

    def running(self):
        """Started, healthy allocations that accept new work, soonest end first."""
        healthy = [
            a
            for a in self.active().values()
            if a["status"] == "running"
            and a.get("deadline")
            and a.get("heartbeat")
            and self.clock() - a["heartbeat"] <= self.config["heartbeat_timeout"]
            and not (Path(a["dir"]) / "drain").exists()
        ]
        return sorted(healthy, key=lambda a: a["deadline"])

    def fits(self, alloc, need, estimate):
        return need <= self.free_gpus(alloc) and (
            estimate + self.config["start_margin"] <= self.remaining(alloc)
        )

    def read(self, alloc, folder, key):
        try:
            return json.loads((Path(alloc["dir"]) / folder / f"{key}.json").read_text())
        except (OSError, ValueError):
            return None

    # ---- allocations -----------------------------------------------------

    def _python(self):
        return self.config["python"] or os.path.realpath(sys.executable)

    def render(self, alloc_id):
        context = dict(
            self.config["vars"],
            gpus=self.config["gpus"],
            time=self.config["time"],
            allocation=alloc_id,
        )
        directory = self.root / alloc_id
        python = self._python()
        job_name = self.config["job_name"].format_map(context)
        from dpgen.dispatcher.fp_slurm import _shell_quote as quote

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            "#SBATCH --output=slurm-%j.out",
            f"#SBATCH --chdir={directory}",
            f"#SBATCH --signal=B:USR1@{int(self.config['warn_before_end'])}",
        ]
        lines += [f"#SBATCH {d.format_map(context)}" for d in self.config["directives"]]
        lines += [
            "",
            f"# dpgen allocation {alloc_id}: {self.config['gpus']} GPU(s) for "
            f"{self.config['time']}. Work is handed to it only after it starts;",
            "# each job gets its own GPU(s) through CUDA_VISIBLE_DEVICES.",
            "",
        ]
        lines += [line.format_map(context) for line in self.config["prepend"]]
        lines += [
            f"cd {quote(str(directory))}",
            f"exec {quote(python)} fp_pilot.py --gpus {self.config['gpus']} "
            f"--time-limit {int(self.config['time_seconds'])} "
            f"--idle-exit {int(self.config['idle_exit'])} --python {quote(python)}",
            "",
        ]
        return job_name, "\n".join(lines)

    def submit_allocation(self, reason, kind):
        alloc_id = f"A{len(self.allocs) + 1:02d}"
        directory = self.root / alloc_id
        job_name, script = self.render(alloc_id)
        alloc = {
            "id": alloc_id,
            "job_id": None,
            "job_name": job_name,
            "status": "submitting",
            "gpus": self.config["gpus"],
            "time_limit": self.config["time_seconds"],
            "submitted_at": self.clock(),
            "dir": str(directory),
            "assigned": {},
            "requested_for": kind,
            "reason": reason,
        }
        self.allocs[alloc_id] = alloc
        if self.dry_run:
            self.scripts[alloc_id] = script
        else:
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fp_pilot.__file__, directory / "fp_pilot.py")
            (directory / "alloc.sbatch").write_text(script)
        self.save()
        from dpgen.dispatcher.fp_slurm import SubmitRejected, SubmitUncertain

        try:
            job_id = self.slurm.submit("alloc.sbatch", cwd=str(directory))
        except SubmitRejected as exc:
            alloc.update(status="rejected", reason=f"sbatch rejected: {exc}")
            self.state["blocked_until"] = self.clock() + 600
            self.event(
                f"allocation {alloc_id} rejected by sbatch: {exc}; retrying in 10 min",
                "error",
            )
            return
        except SubmitUncertain as exc:
            self.event(
                f"allocation {alloc_id}: sbatch outcome unknown ({exc})", "warning"
            )
            return
        alloc.update(job_id=job_id, status="queued")
        self.event(
            f"requested allocation {alloc_id} as job {job_id} "
            f"({self.config['gpus']} GPU × {self.config['time']}) for {kind}: {reason}"
        )

    def _resolve_submitting(self, alloc, queue):
        from dpgen.dispatcher.fp_slurm import _same_path

        directory = alloc["dir"]
        found = [j for j, job in queue.items() if _same_path(job["workdir"], directory)]
        if not found:
            try:
                found = self.slurm.find(directory, alloc["submitted_at"] - 300)
            except Exception:  # noqa: BLE001 — accounting may lag; retry next pass
                return
        if found:
            alloc.update(job_id=found[-1], status="queued")
            self.event(
                f"allocation {alloc['id']}: found job {found[-1]} after uncertain sbatch"
            )
        elif self.clock() - alloc["submitted_at"] > self.config["accounting_grace"]:
            alloc.update(status="not_submitted")
            self.event(
                f"allocation {alloc['id']}: no job was created; it will be requested again"
            )

    def update(self, queue):
        """Refresh every active allocation from the queue and its heartbeat."""
        for alloc in list(self.active().values()):
            if alloc["status"] == "submitting":
                self._resolve_submitting(alloc, queue)
                continue
            job = queue.get(str(alloc["job_id"]))
            status = {}
            try:
                status = json.loads((Path(alloc["dir"]) / "status.json").read_text())
            except (OSError, ValueError):
                pass
            if status:
                alloc.update(
                    started=status.get("started"),
                    deadline=status.get("deadline"),
                    heartbeat=status.get("time"),
                    node=status.get("node"),
                )
                self.state["failed_starts"] = 0
            if job is not None:
                before = alloc["status"]
                alloc["status"] = "queued" if job["state"] == "PENDING" else "running"
                if before != "running" and alloc["status"] == "running":
                    self.event(
                        f"allocation {alloc['id']} (job {alloc['job_id']}) started"
                    )
                    waited = (alloc.get("started") or self.clock()) - alloc[
                        "submitted_at"
                    ]
                    self._milestone(
                        "node_started",
                        f"node {alloc['id']} (job {alloc['job_id']}) started on "
                        f"{alloc.get('node') or '?'} after {progress.duration(waited)} "
                        f"in the queue ({alloc['gpus']} GPU × {hours(alloc['time_limit'])})",
                        alloc,
                        waited=round(waited),
                    )
            else:
                alloc.update(status="ended", ended=self.clock())
                self.event(f"allocation {alloc['id']} (job {alloc['job_id']}) ended")
                self._check_failed_start(alloc)
                if alloc.get("started"):
                    ran = self.clock() - alloc["started"]
                    self._milestone(
                        "node_ended",
                        f"node {alloc['id']} (job {alloc['job_id']}) ended after "
                        f"{progress.duration(ran)}",
                        alloc,
                        ran=round(ran),
                    )

    def _check_failed_start(self, alloc):
        """A node that ran but whose pilot never checked in: stop after two."""
        output = Path(alloc["dir"]) / f"slurm-{alloc['job_id']}.out"
        if alloc.get("heartbeat") or alloc.get("cancel_requested"):
            return
        if not output.exists():
            return  # never started (e.g. cancelled while pending)
        count = self.state.get("failed_starts", 0) + 1
        self.state["failed_starts"] = count
        self.event(
            f"allocation {alloc['id']} (job {alloc['job_id']}) ended before its pilot "
            f"started; see {output}",
            "error",
        )
        if count >= 2:
            self.state["halted"] = (
                f"{count} nodes in a row ended before the pilot started (last: {output}). "
                "Fix the cause (often pool.python or pool.prepend), then run "
                "`dpgen pool --retry`."
            )
            self.event(
                f"no more nodes will be requested: {self.state['halted']}", "error"
            )

    def _milestone(self, event, message, alloc, **data):
        if not self.dry_run:
            progress.summary(
                self.workdir,
                event,
                message,
                now=self.clock(),
                allocation=alloc["id"],
                job_id=alloc.get("job_id"),
                node=alloc.get("node"),
                **data,
            )

    def live_text(self):
        """Nodes in one short phrase, e.g. ``A01 4/4 GPU 18.2 h left``."""
        parts = []
        for alloc in sorted(self.active().values(), key=lambda a: a["id"]):
            if alloc["status"] == "running":
                parts.append(
                    f"{alloc['id']} {self.used_gpus(alloc)}/{alloc['gpus']} GPU "
                    f"{hours(max(0, self.remaining(alloc)))} left"
                )
            else:
                parts.append(f"{alloc['id']} {alloc['status']}")
        return "nodes " + (", ".join(parts) if parts else "none")

    def end_all(self, reason):
        for alloc in self.active().values():
            alloc.update(status="ended", ended=self.clock(), reason=reason)

    # ---- work ------------------------------------------------------------

    def assign(
        self, alloc, key, *, cwd, argv, ngpus, estimate, owner, log=None, **extra
    ):
        alloc.setdefault("assigned", {})[key] = {
            "ngpus": ngpus,
            "estimate": round(estimate),
            "owner": owner,
            "assigned_at": self.clock(),
            **extra,
        }
        if not self.dry_run:
            (Path(alloc["dir"]) / "inbox").mkdir(parents=True, exist_ok=True)
            fp_pilot.atomic_json(
                Path(alloc["dir"]) / "inbox" / f"{key}.json",
                {
                    "name": key,
                    "task_dir": str(cwd),
                    "argv": list(argv),
                    "log": log,
                    "ngpus": ngpus,
                    "estimate": round(estimate),
                    "assigned_at": self.clock(),
                    **extra,
                },
            )

    def release(self, alloc, key, suffix=""):
        """Forget a finished assignment; keep its record under processed/."""
        alloc.get("assigned", {}).pop(key, None)
        if self.dry_run:
            return
        source = Path(alloc["dir"]) / "finished" / f"{key}.json"
        if source.exists():
            done = Path(alloc["dir"]) / "processed"
            done.mkdir(exist_ok=True)
            os.replace(source, done / f"{key}{suffix}.json")

    def cancel(self, alloc, key, intent):
        if not self.dry_run:
            (Path(alloc["dir"]) / "cancel").mkdir(parents=True, exist_ok=True)
            fp_pilot.atomic_json(
                Path(alloc["dir"]) / "cancel" / f"{key}.json", {"intent": intent}
            )

    def balance(self, waiting_gpus, waiting_count, kind, shortest=0.0):
        """Request allocations for queued work; release the ones not needed.

        A running allocation too close to its end to start even the shortest
        waiting job no longer counts toward ``max_allocations``: its successor
        is requested while it finishes what it runs. With nothing waiting, the
        ``keep`` allocations with the most time left stay for the next stage
        and the others are released (see :meth:`_release_unneeded`).
        """
        active = self.active()
        if not waiting_count:
            self._release_unneeded(active)
            self._renew(active, kind)
            return
        for alloc in active.values():
            flag = Path(alloc["dir"]) / "drain"
            too_late = (
                alloc["status"] == "running"
                and self.remaining(alloc) < shortest + self.config["start_margin"]
            )
            if too_late and not alloc.get("assigned"):
                # Idle, and too close to its end for any waiting job: let it go
                # now instead of after idle_exit.
                if not flag.exists():
                    if not self.dry_run:
                        flag.write_text("too little time left for the waiting work\n")
                    self.event(
                        f"allocation {alloc['id']} has {hours(max(0, self.remaining(alloc)))} "
                        "left, too little for any waiting job; asked it to exit"
                    )
            elif flag.exists() and not self.dry_run:
                flag.unlink()
        if self.clock() < self.state.get("blocked_until", 0) or self.state.get(
            "halted"
        ):
            return
        spare = sum(
            self.free_gpus(a) for a in active.values() if a["status"] != "running"
        )
        useful = [
            a
            for a in active.values()
            if a["status"] != "running"
            or self.remaining(a) >= shortest + self.config["start_margin"]
        ]
        if len(useful) < self.limit(kind) and waiting_gpus > spare:
            self.submit_allocation(
                f"{waiting_count} job(s) waiting, {waiting_gpus} GPU(s) needed", kind
            )
            return
        self._renew(active, kind)

    def keepers(self, active=None):
        """IDs of the allocations kept between stages: most time left first."""
        active = self.active() if active is None else active
        running = sorted(
            (a for a in active.values() if a["status"] == "running"),
            key=lambda a: -(a.get("deadline") or 0),
        )
        waiting = sorted(
            (a for a in active.values() if a["status"] != "running"),
            key=lambda a: a["submitted_at"],
        )
        return {a["id"] for a in (running + waiting)[: self.config["keep"]]}

    def _release_unneeded(self, active):
        """Nothing is waiting: keep ``keep`` allocations, let the others go.

        A running one is asked to exit once idle; one still in the queue is
        cancelled while pending. Allocations with work on them are never touched.
        """
        kept = self.keepers(active)
        for alloc in active.values():
            if alloc["id"] in kept or alloc.get("assigned"):
                continue
            if alloc["status"] == "queued" and alloc.get("job_id"):
                if alloc.get("cancel_requested"):
                    continue
                try:
                    self.slurm.cancel(alloc["job_id"], pending_only=True)
                    alloc["cancel_requested"] = self.clock()
                    self.event(
                        f"allocation {alloc['id']} no longer needed; cancelling it while pending"
                    )
                except Exception as exc:  # noqa: BLE001
                    self.event(
                        f"could not cancel allocation {alloc['id']}: {exc}", "warning"
                    )
            elif alloc["status"] == "running":
                flag = Path(alloc["dir"]) / "drain"
                if not flag.exists():
                    if not self.dry_run:
                        flag.write_text("no work left\n")
                    self.event(
                        f"allocation {alloc['id']} has no work left; asked it to exit"
                        + (" (another node is kept for the next stage)" if kept else "")
                    )

    def _renew(self, active, kind):
        """Queue the kept node's successor ``renew_before`` ahead of its end.

        Only while work is running or waiting, and only when no other node is
        queued or has more than ``renew_before`` left, so the workflow does not
        fall back into the queue when the node it relies on ends.
        """
        lead = self.config["renew_before_seconds"]
        if not lead or not active or self.config["keep"] < 1:
            return
        busy = any(a.get("assigned") for a in active.values()) or any(
            i["status"] in ("waiting", "assigned", "running")
            for i in self.items.values()
        )
        if not busy:
            return
        if any(
            a["status"] != "running" or self.remaining(a) > lead
            for a in active.values()
        ):
            return
        if self.clock() < self.state.get("blocked_until", 0) or self.state.get(
            "halted"
        ):
            return
        ending = min(active.values(), key=lambda a: self.remaining(a))
        self.submit_allocation(
            f"renewal: {ending['id']} ends in {hours(max(0, self.remaining(ending)))}",
            kind or ending.get("requested_for") or "other",
        )

    # ---- generic jobs (training, exploration, ...) ----------------------

    def estimate(self, kind):
        """Return (seconds, basis) for a job of ``kind``, safety factor included."""
        ceiling = self.config["time_seconds"] - self.config["start_margin"]
        past = self.state["history"].get(kind, [])
        if past:
            value = statistics.median(past) * self.config["safety_factor"]
            basis = f"median of {len(past)} earlier {kind} job(s)"
        elif kind in self.config["runtime_seconds"]:
            value = self.config["runtime_seconds"][kind]
            basis = f"pool.runtime.{kind}"
        else:
            return 0.0, "no estimate (starts whenever a GPU is free)"
        if value > ceiling:
            return ceiling, basis + "; capped to one allocation"
        return value, basis

    def enqueue(self, key, *, kind, cwd, argv, ngpus, log=None, label=None):
        self.items[key] = {
            "kind": kind,
            "cwd": str(cwd),
            "argv": list(argv),
            "ngpus": int(ngpus),
            "log": log,
            "label": label or key,
            "status": "waiting",
            "submitted_at": self.clock(),
        }

    def drop(self, key, reason="cancelled"):
        item = self.items.get(key)
        if item is None:
            return
        alloc = self.allocs.get(item.get("allocation") or "")
        if item["status"] in ("assigned", "running") and alloc is not None:
            self.cancel(alloc, key, reason)
        item.update(status="cancelled", reason=reason, ended=self.clock())

    def _settle_item(self, key, item):
        alloc = self.allocs.get(item.get("allocation") or "")
        if alloc is None:
            item.update(status="waiting", allocation=None)
            return
        started = self.read(alloc, "running", key)
        if started and item["status"] == "assigned":
            item.update(
                status="running",
                started=started.get("start"),
                gpus=started.get("gpus"),
                node=started.get("node"),
            )
        record = self.read(alloc, "finished", key)
        if record is not None:
            self.release(alloc, key)
            if record.get("not_started") and not record.get("cancelled"):
                item.update(
                    status="waiting", allocation=None, reason=record.get("reason")
                )
                return
            if record.get("interrupted") or record.get("cancelled"):
                item.update(
                    status="interrupted",
                    ended=self.clock(),
                    reason="cancelled"
                    if record.get("cancelled")
                    else f"allocation stopped (signal {record.get('signal')})",
                )
                return
            code = record.get("exit_code")
            item.update(status="finished", exit_code=code, ended=self.clock())
            if code == 0 and record.get("start") and record.get("end"):
                past = self.state["history"].setdefault(item["kind"], [])
                past.append(round(record["end"] - record["start"], 1))
                del past[:-HISTORY_KEEP]
            return
        if alloc["status"] not in ACTIVE:
            alloc.get("assigned", {}).pop(key, None)
            if item.get("started") or started:
                item.update(
                    status="interrupted",
                    ended=self.clock(),
                    reason=f"allocation {alloc['id']} ended while it ran",
                )
            else:
                item.update(status="waiting", allocation=None)

    def step_items(self, queue):
        """One pass for generic jobs: settle, place on started nodes, request more."""
        self.update(queue)
        for key, item in self.items.items():
            if item["status"] in ("assigned", "running"):
                self._settle_item(key, item)
        waiting = sorted(
            (k for k, i in self.items.items() if i["status"] == "waiting"),
            key=lambda k: (-self.items[k]["ngpus"], self.items[k]["submitted_at"]),
        )
        left = []
        for key in waiting:
            item = self.items[key]
            estimate, basis = self.estimate(item["kind"])
            if item["ngpus"] > self.config["gpus"]:
                item.update(
                    status="interrupted",
                    reason=f"needs {item['ngpus']} GPUs; allocations have {self.config['gpus']}",
                )
                self.event(f"{item['label']}: {item['reason']}", "error")
                continue
            # Most time left first: the node that is kept between stages.
            nodes = sorted(self.running(), key=lambda a: -a["deadline"])
            target = next(
                (a for a in nodes if self.fits(a, item["ngpus"], estimate)),
                None,
            )
            if target is None:
                left.append(key)
                continue
            self.assign(
                target,
                key,
                cwd=item["cwd"],
                argv=item["argv"],
                ngpus=item["ngpus"],
                estimate=estimate,
                owner=item["kind"],
                log=item["log"],
            )
            item.update(status="assigned", allocation=target["id"])
            self.event(
                f"{item['label']} assigned to allocation {target['id']} "
                f"(job {target['job_id']}): {item['ngpus']} GPU, estimate "
                f"{hours(estimate)} ({basis})"
            )
        if left:
            kinds = [self.items[k]["kind"] for k in left]
            kind = max(set(kinds), key=kinds.count)
        else:
            kind = None
        shortest = min(
            (self.estimate(self.items[k]["kind"])[0] for k in left), default=0
        )
        self.balance(
            sum(self.items[k]["ngpus"] for k in left), len(left), kind, shortest
        )
        self.state["last_step"] = self.clock()
        self.save()
        if not self.dry_run:
            counts = {}
            for item in self.items.values():
                if item["status"] in ("waiting", "assigned", "running"):
                    label = f"{item['kind']} {item['status']}"
                    counts[label] = counts.get(label, 0) + 1
            jobs = ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "no jobs"
            progress.live(self.workdir, f"pool | {jobs} | {self.live_text()}")


def summary_rows(allocs, now=None):
    now = time.time() if now is None else now
    rows = []
    for alloc_id, alloc in sorted(allocs.items()):
        remaining = (alloc.get("deadline") or 0) - now
        assigned = alloc.get("assigned", {})
        rows.append(
            {
                "allocation": alloc_id,
                "job_id": alloc.get("job_id") or "-",
                "status": alloc["status"],
                "node": alloc.get("node") or "-",
                "gpus": f"{sum(i['ngpus'] for i in assigned.values())}/{alloc['gpus']} busy",
                "time_left": hours(remaining)
                if alloc["status"] == "running" and remaining > 0
                else "-",
                "tasks": ", ".join(sorted(assigned)) or "-",
            }
        )
    return rows


# --------------------------------------------------------------------------
# command line: dpgen pool


def command(args, out=sys.stdout):
    root = pool_root(args.WORKDIR or ".")
    state = read_state(root)
    if state is None:
        raise SystemExit(f"No node pool ({STATE}) under {root}")
    if getattr(args, "retry", False):
        state.pop("halted", None)
        state["failed_starts"] = 0
        fp_pilot.atomic_json(root / STATE, state)
        print(
            "Node requests allowed again; dpgen requests nodes at its next check.",
            file=out,
        )
        return
    if state.get("halted"):
        print(f"NODE REQUESTS STOPPED: {state['halted']}\n", file=out)
    if args.drain:
        for alloc in state["allocations"].values():
            if alloc["status"] in ACTIVE and Path(alloc["dir"]).is_dir():
                (Path(alloc["dir"]) / "drain").write_text("drained by user\n")
                print(f"{alloc['id']} will exit once its running jobs finish", file=out)
        return
    print(
        "ALLOCATION  JOB_ID      STATUS     NODE        GPUS        LEFT     JOBS",
        file=out,
    )
    for row in summary_rows(state["allocations"]):
        print(
            f"{row['allocation']:<10}  {row['job_id']:<10}  {row['status']:<9}  "
            f"{row['node']:<10}  {row['gpus']:<10}  {row['time_left']:<7}  {row['tasks']}",
            file=out,
        )
    items = state.get("items", {})
    shown = [
        (key, item)
        for key, item in items.items()
        if args.all or item["status"] in ("waiting", "assigned", "running")
    ]
    if shown:
        print("\nJOB" + " " * 34 + "STATUS       ALLOC  GPUS  WHAT", file=out)
        for key, item in sorted(shown, key=lambda pair: pair[1]["submitted_at"]):
            print(
                f"{key:<37} {item['status']:<12} {item.get('allocation') or '-':<6} "
                f"{item['ngpus']:<5} {item.get('label', '')}",
                file=out,
            )
    counts = {}
    for item in items.values():
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    if counts:
        print(
            "\njobs: " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
            file=out,
        )
    for kind, past in sorted(state.get("history", {}).items()):
        if past:
            print(
                f"typical {kind} job: {hours(statistics.median(past))} "
                f"(median of {len(past)})",
                file=out,
            )


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "pool",
        help="Show the node pool (long allocations shared by training, "
        "exploration and FP) or ask its nodes to exit.",
    )
    parser.add_argument("WORKDIR", nargs="?", help="dpgen run directory (default: .)")
    parser.add_argument(
        "--all", action="store_true", help="also list finished and cancelled jobs"
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="allow node requests again after they were stopped by failed starts",
    )
    parser.add_argument(
        "--drain",
        action="store_true",
        help="ask every allocation to exit when its running jobs finish",
    )
    parser.set_defaults(func=command)
