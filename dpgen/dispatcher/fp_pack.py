"""Pack FP single points into long allocations ("pilot" jobs).

Instead of one Slurm job per single point, the controller keeps a few long
allocations (for example 1 node x 4 GPUs x 24 h) and hands tasks to an
allocation only once it is running. Inside each allocation ``fp_pilot.py`` runs
tasks side by side on the GPUs. A task starts only when its estimated runtime
fits the allocation's remaining time; the estimate comes from
:class:`RuntimeModel`, which learns from every finished task.

All of the per-task guarantees of :mod:`dpgen.dispatcher.fp_slurm` still hold:
outputs are validated before a task is done, validated results are frozen,
failed attempts are archived, and a task is never assigned twice.
"""

import json
import math
import os
import re
import shutil
import statistics
import time
from pathlib import Path

from dpgen.dispatcher import fp_pilot
from dpgen.dispatcher import fp_recovery_worker as worker
from dpgen.dispatcher import fp_vasp_monitor as vasp_monitor

ALLOCATIONS = "allocations"
RUNTIME_HISTORY = ".dpgen-fp-runtime.jsonl"
MAX_INTERRUPTIONS = 5

PACK_DEFAULTS = {
    "enabled": False,
    "gpus": 4,
    "time": "24:00:00",
    "max_allocations": 1,
    "directives": [],
    "idle_exit": 900,
    "safety_factor": 1.3,
    "default_runtime": "02:00:00",
    "start_margin": 600,
    "heartbeat_timeout": 600,
    "warn_before_end": 300,
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
        hours, minutes, secs = (parts + [0, 0])[:3]
    elif len(parts) == 3:
        hours, minutes, secs = parts
    elif len(parts) == 2:
        hours, minutes, secs = 0, parts[0], parts[1]
    else:
        hours, minutes, secs = 0, parts[0], 0
    return float(int(days or 0) * 86400 + hours * 3600 + minutes * 60 + secs)


def hours(value):
    return f"{value / 3600:.1f} h"


def pack_options(value, base_vars):
    value = {} if value is None else value
    if not isinstance(value, dict) or value.keys() - PACK_DEFAULTS.keys():
        unknown = (
            sorted(set(value) - set(PACK_DEFAULTS))
            if isinstance(value, dict)
            else value
        )
        raise ValueError(f"Invalid fp.slurm.pack options: {unknown}")
    config = dict(PACK_DEFAULTS, **value)
    if not isinstance(config["enabled"], bool):
        raise ValueError("fp.slurm.pack.enabled must be boolean")
    if not config["enabled"]:
        return config
    for key in ("gpus", "max_allocations"):
        if (
            isinstance(config[key], bool)
            or not isinstance(config[key], int)
            or config[key] < 1
        ):
            raise ValueError(f"fp.slurm.pack.{key} must be a positive integer")
    for key in ("time", "default_runtime"):
        try:
            config[f"{key}_seconds"] = seconds(config[key])
        except ValueError as exc:
            raise ValueError(f"fp.slurm.pack.{key}: {exc}") from exc
    for key in (
        "idle_exit",
        "safety_factor",
        "start_margin",
        "heartbeat_timeout",
        "warn_before_end",
    ):
        number = config[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number < 0
        ):
            raise ValueError(f"fp.slurm.pack.{key} must be a non-negative number")
    if config["safety_factor"] < 1:
        raise ValueError("fp.slurm.pack.safety_factor must be at least 1")
    if not isinstance(config["directives"], list) or not all(
        isinstance(item, str) and item.startswith("-") for item in config["directives"]
    ):
        raise ValueError("fp.slurm.pack.directives must be a list of sbatch options")
    for directive in config["directives"]:
        if re.match(r"--(?:chdir|output|job-name|signal)\b|-[DoJ]\b", directive):
            raise ValueError(f"{directive!r} is set by the controller; remove it")
    names = set(base_vars) | {"gpus", "time", "iter", "allocation"}
    from dpgen.dispatcher.fp_slurm import _placeholders

    for directive in config["directives"]:
        missing = _placeholders(directive) - names
        if missing:
            raise ValueError(
                f"fp.slurm.pack.directives uses undefined variable(s) {sorted(missing)}"
            )
    return config


class RuntimeModel:
    """Runtime estimates from past single points, per site and GPU count.

    With three or more results at different sizes for the same site and GPU
    count, it fits ``t = a * natoms**b``. With fewer, it scales the nearest
    results by ``natoms**2.5``. With none, it uses the configured default.
    """

    EXPONENT = 2.5

    def __init__(self, path, default_seconds, site="default"):
        self.path = Path(path)
        self.default = float(default_seconds)
        self.site = site
        self.rows = []
        try:
            for line in self.path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("seconds", 0) > 0 and row.get("natoms", 0) > 0:
                    self.rows.append(row)
        except OSError:
            pass

    def record(self, natoms, ngpus, elapsed, task, dry_run=False):
        if not elapsed or elapsed <= 0 or natoms <= 0:
            return
        row = {
            "site": self.site,
            "ngpus": int(ngpus),
            "natoms": int(natoms),
            "seconds": round(float(elapsed), 1),
            "task": task,
            "time": time.time(),
        }
        self.rows.append(row)
        if not dry_run:
            with self.path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    def estimate(self, natoms, ngpus):
        """Return (seconds, basis, confident) for one single point.

        ``confident`` is False when the value is extrapolated far from the
        recorded sizes or GPU counts, or is the configured default.
        """
        same = [
            r
            for r in self.rows
            if r.get("site") == self.site and r.get("ngpus") == ngpus
        ]
        if len(same) >= 3 and len({r["natoms"] for r in same}) >= 2:
            xs = [math.log(r["natoms"]) for r in same]
            ys = [math.log(r["seconds"]) for r in same]
            mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
            spread = sum((x - mean_x) ** 2 for x in xs)
            slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / spread
            slope = min(max(slope, 1.0), 3.5)
            value = math.exp(mean_y + slope * (math.log(natoms) - mean_x))
            sizes = [r["natoms"] for r in same]
            confident = min(sizes) / 1.5 <= natoms <= max(sizes) * 1.5
            return value, f"fit of {len(same)} runs (t ∝ N^{slope:.1f})", confident
        if same:
            scaled = [
                r["seconds"] * (natoms / r["natoms"]) ** self.EXPONENT for r in same
            ]
            confident = any(1 / 1.5 <= natoms / r["natoms"] <= 1.5 for r in same)
            basis = f"scaled from {len(same)} run(s)"
            return statistics.median(scaled), basis, confident
        other = [r for r in self.rows if r.get("site") == self.site]
        if other:
            scaled = [
                r["seconds"]
                * (natoms / r["natoms"]) ** self.EXPONENT
                * (r["ngpus"] / ngpus) ** 0.7
                for r in other
            ]
            basis = f"scaled from {len(other)} run(s) on other GPU counts"
            return statistics.median(scaled), basis, False
        return self.default, "default_runtime (no history yet)", False


def elapsed_of(task_dir, accounting=None):
    """VASP wall time of the last attempt, from the worker's record or Slurm."""
    try:
        report = json.loads((Path(task_dir) / "fp-attempt.json").read_text())
        if report.get("started") and report.get("finished"):
            return float(report["finished"]) - float(report["started"])
    except (OSError, ValueError, TypeError):
        pass
    text = (accounting or {}).get("elapsed")
    if isinstance(text, (int, float)):
        return float(text)
    if text:
        try:
            return seconds(text)
        except ValueError:
            return None
    return None


class PackManager:
    """Allocation lifecycle and task assignment for one FP stage."""

    def __init__(self, campaign, config):
        self.c = campaign
        self.config = config
        self.allocs = campaign.state.setdefault("allocations", {})
        self.dir = campaign.root / ALLOCATIONS

    # ---- helpers ---------------------------------------------------------

    def _now(self):
        return self.c.clock()

    def _alloc_dir(self, alloc_id):
        return self.dir / alloc_id

    def _read(self, path):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return None

    def ngpus(self, name):
        value = self.c.resolve_vars(name).get("ngpus", 1)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

    def estimate(self, name):
        """Return (seconds, basis, firm) for a task, including the safety factor.

        ``firm`` means the estimate is trusted enough to hold a task that would
        not fit any allocation. Uncertain estimates are capped at what a fresh
        allocation can hold, so the task still runs and teaches the model.
        """
        entry = self.c.state["tasks"][name]
        need = self.ngpus(name)
        value, basis, confident = self.c.runtime.estimate(entry["natoms"], need)
        value *= self.config["safety_factor"]
        ceiling = self.config["time_seconds"] - self.config["start_margin"]
        if not confident and value > ceiling:
            value, basis = ceiling, basis + "; uncertain, capped to one allocation"
        floor = entry.get("min_runtime")
        if floor and floor * 1.2 > value:
            return floor * 1.2, "at least what an interrupted attempt already ran", True
        return value, basis, confident

    def active(self):
        return {
            alloc_id: alloc
            for alloc_id, alloc in self.allocs.items()
            if alloc["status"] in ("submitting", "queued", "running")
        }

    def assigned_gpus(self, alloc):
        return sum(item["ngpus"] for item in alloc.get("assigned", {}).values())

    def assignable(self):
        tasks = self.c.state["tasks"]
        names = []
        for name, entry in tasks.items():
            if entry["status"] != "pending" or entry.get("submitting"):
                continue
            if entry.get("hold"):
                entry.update(status="held", reason="held by user")
                continue
            if self._now() < entry.get("retry_after", 0):
                continue
            names.append(name)
        return names

    # ---- allocations -----------------------------------------------------

    def render(self, alloc_id):
        context = dict(
            self.c.config["vars"],
            gpus=self.config["gpus"],
            time=self.config["time"],
            iter=self.c.iter_label,
            allocation=alloc_id,
        )
        directory = self._alloc_dir(alloc_id)
        python = self.c.config["python"] or os.path.realpath(
            __import__("sys").executable
        )
        job_name = f"{self.c.iter_label}-fp-pack-{alloc_id}"
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
            f"# dpgen FP {self.c.iter_label} allocation {alloc_id}: "
            f"{self.config['gpus']} GPU(s) for {self.config['time']}; runs several",
            "# single points side by side. Tasks are assigned only after it starts.",
            "",
        ]
        lines += [line.format_map(context) for line in self.c.config["prepend"]]
        lines += [
            f"cd {self.c._quote(str(directory))}",
            f"exec {self.c._quote(python)} fp_pilot.py --gpus {self.config['gpus']} "
            f"--time-limit {int(self.config['time_seconds'])} "
            f"--idle-exit {int(self.config['idle_exit'])} --python {self.c._quote(python)}",
            "",
        ]
        return job_name, "\n".join(lines)

    def submit_allocation(self, reason):
        number = len(self.allocs) + 1
        alloc_id = f"A{number:02d}"
        directory = self._alloc_dir(alloc_id)
        job_name, script = self.render(alloc_id)
        alloc = {
            "id": alloc_id,
            "job_id": None,
            "job_name": job_name,
            "status": "submitting",
            "gpus": self.config["gpus"],
            "time_limit": self.config["time_seconds"],
            "submitted_at": self._now(),
            "dir": str(directory),
            "assigned": {},
            "reason": reason,
        }
        self.allocs[alloc_id] = alloc
        if self.c.dry_run:
            self.c.alloc_scripts[alloc_id] = script
        else:
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fp_pilot.__file__, directory / "fp_pilot.py")
            (directory / "alloc.sbatch").write_text(script)
        self.c.save()
        from dpgen.dispatcher.fp_slurm import SubmitRejected, SubmitUncertain

        try:
            job_id = self.c.slurm.submit("alloc.sbatch", cwd=str(directory))
        except SubmitRejected as exc:
            alloc.update(status="rejected", reason=f"sbatch rejected: {exc}")
            self.c.state["pack_blocked_until"] = self._now() + 600
            self.c.event(
                None,
                f"allocation {alloc_id} rejected by sbatch: {exc}; retrying in 10 min",
                "error",
            )
            return
        except SubmitUncertain as exc:
            self.c.event(
                None,
                f"allocation {alloc_id}: sbatch outcome unknown ({exc})",
                "warning",
            )
            return
        alloc.update(job_id=job_id, status="queued")
        self.c.event(
            None,
            f"requested allocation {alloc_id} as job {job_id} "
            f"({self.config['gpus']} GPU × {self.config['time']}): {reason}",
        )

    def _resolve_submitting(self, alloc, queue):
        directory = alloc["dir"]
        found = [
            j
            for j, job in queue.items()
            if self.c._same_path(job["workdir"], directory)
        ]
        if not found:
            try:
                found = self.c.slurm.find(directory, alloc["submitted_at"] - 300)
            except Exception:  # noqa: BLE001 — accounting may lag; retry next pass
                return
        if found:
            alloc.update(job_id=found[-1], status="queued")
            self.c.event(
                None,
                f"allocation {alloc['id']}: found job {found[-1]} after uncertain sbatch",
            )
        elif self._now() - alloc["submitted_at"] > self.c.config["accounting_grace"]:
            alloc.update(status="not_submitted")
            self.c.event(
                None,
                f"allocation {alloc['id']}: no job was created; it will be requested again",
            )

    def update(self, queue):
        for alloc in list(self.active().values()):
            if alloc["status"] == "submitting":
                self._resolve_submitting(alloc, queue)
                continue
            job = queue.get(str(alloc["job_id"]))
            status = self._read(Path(alloc["dir"]) / "status.json") or {}
            if status:
                alloc.update(
                    started=status.get("started"),
                    deadline=status.get("deadline"),
                    heartbeat=status.get("time"),
                    node=status.get("node"),
                )
            if job is not None:
                alloc["status"] = "queued" if job["state"] == "PENDING" else "running"
            self.collect(alloc)
            if job is None:
                self.end(alloc)

    def collect(self, alloc):
        directory = Path(alloc["dir"])
        tasks = self.c.state["tasks"]
        for path in sorted((directory / "running").glob("*.json")):
            name = path.stem
            entry = tasks.get(name)
            record = self._read(path)
            if not entry or not record or name not in alloc["assigned"]:
                continue
            attempt = entry["attempts"][-1]
            if attempt.get("started_at") is None:
                attempt.update(
                    started_at=record.get("start"),
                    gpus=record.get("gpus"),
                    node=record.get("node"),
                )
                entry.update(status="running", reason="")
                self.c.event(
                    name,
                    f"started in allocation {alloc['id']} (job {alloc['job_id']}) "
                    f"on GPU {','.join(record.get('gpus', []))}",
                )
        for path in sorted((directory / "finished").glob("*.json")):
            name = path.stem
            record = self._read(path)
            if record is None or name not in alloc["assigned"]:
                continue
            item = alloc["assigned"].pop(name)
            if not self.c.dry_run:
                done = directory / "processed"
                done.mkdir(exist_ok=True)
                os.replace(path, done / f"{name}.a{item['attempt']}.json")
            self.finish_task(name, alloc, record)

    def finish_task(self, name, alloc, record):
        entry = self.c.state["tasks"][name]
        attempt = entry["attempts"][-1]
        elapsed = (
            (record.get("end") or 0) - (record.get("start") or 0)
            if record.get("start")
            else None
        )
        if record.get("not_started"):
            attempt.update(
                category="not_started",
                reason=record.get("reason", "not started"),
                ended=self._now(),
            )
            intent = attempt.get("cancel_intent")
            if intent == "discard":
                entry.update(
                    status="discarded", reason=entry.get("discard", "discarded")
                )
            elif intent == "hold":
                entry.update(status="held", reason="held by user")
            else:
                entry.update(
                    status="pending", retry_after=0, reason=record.get("reason", "")
                )
            self.c.event(
                name,
                f"not started in allocation {alloc['id']}: {record.get('reason', intent)}",
            )
            return
        if record.get("interrupted"):
            entry["interruptions"] = entry.get("interruptions", 0) + 1
            if elapsed:
                entry["min_runtime"] = max(entry.get("min_runtime", 0), elapsed)
            task_dir = self.c.root / name
            if (task_dir / "OUTCAR").exists() and self.c.validate(task_dir)[0]:
                self.c.finish(
                    name,
                    {
                        "state": "COMPLETED",
                        "elapsed": elapsed,
                        "node": record.get("node"),
                    },
                )
                return
            attempt.update(
                category="preemption",
                reason=f"allocation stopped (signal {record.get('signal')}) after {hours(elapsed or 0)}",
                ended=self._now(),
            )
            if entry["interruptions"] > MAX_INTERRUPTIONS:
                entry.update(
                    status="held",
                    reason=f"interrupted {entry['interruptions']} times; check its runtime",
                )
                self.c.event(
                    name, f"interrupted {entry['interruptions']} times; held", "error"
                )
                return
            entry.update(status="pending", retry_after=0, reason=attempt["reason"])
            self.c.event(
                name,
                f"{attempt['reason']}; back on the list (no retry used)",
                "warning",
            )
            return
        code = record.get("exit_code")
        state = (
            "CANCELLED"
            if record.get("cancelled")
            else ("COMPLETED" if code == 0 else "FAILED")
        )
        self.c.finish(
            name,
            {
                "state": state,
                "exit_code": f"{code}:0" if code is not None else None,
                "node": record.get("node"),
                "elapsed": elapsed,
            },
        )

    def end(self, alloc):
        """The allocation left the queue: settle every task still assigned to it."""
        if alloc["status"] == "ended":
            return
        self.collect(alloc)
        alloc.update(status="ended", ended=self._now())
        for name, item in list(alloc["assigned"].items()):
            entry = self.c.state["tasks"][name]
            attempt = entry["attempts"][-1]
            started = attempt.get("started_at")
            if started:
                elapsed = self._now() - started
                entry["min_runtime"] = max(entry.get("min_runtime", 0), elapsed)
                record = {
                    "interrupted": True,
                    "start": started,
                    "end": self._now(),
                    "signal": "allocation ended",
                }
            else:
                record = {
                    "not_started": True,
                    "reason": f"allocation {alloc['id']} ended before it started",
                }
            alloc["assigned"].pop(name)
            self.finish_task(name, alloc, record)
        self.c.event(None, f"allocation {alloc['id']} (job {alloc['job_id']}) ended")

    # ---- assignment ------------------------------------------------------

    def assign(self, name, alloc, need, estimate, basis):
        entry = self.c.state["tasks"][name]
        task = self.c.root / name
        number = (entry["attempts"][-1]["n"] if entry["attempts"] else 0) + 1
        archived = self.c._archive(name)
        _, command, values, _ = self.c.render(name, number)
        if not self.c.dry_run:
            shutil.copyfile(worker.__file__, task / "fp_recovery_worker.py")
            if self.c.monitor["enabled"]:
                shutil.copyfile(vasp_monitor.__file__, task / "fp_vasp_monitor.py")
            worker.atomic_json(
                task / "fp-recovery.json",
                {
                    "command": command,
                    "bad_gpu_uuids": sorted(self.c._health()["devices"]),
                    "fp_monitor": self.c.monitor,
                    "validated_input_digest": None,
                },
            )
        attempt = {
            "n": number,
            "allocation": alloc["id"],
            "job_id": alloc["job_id"],
            "vars": values,
            "command": command,
            "assigned_at": self._now(),
            "estimate": round(estimate),
            "archived_previous": str(archived.relative_to(task)) if archived else None,
        }
        entry["attempts"].append(attempt)
        entry.update(status="queued", reason=f"assigned to {alloc['id']}")
        alloc["assigned"][name] = {
            "ngpus": need,
            "attempt": number,
            "estimate": estimate,
        }
        if not self.c.dry_run:
            fp_pilot.atomic_json(
                Path(alloc["dir"]) / "inbox" / f"{name}.json",
                {
                    "name": name,
                    "task_dir": str(task),
                    "attempt": number,
                    "ngpus": need,
                    "estimate": round(estimate),
                    "assigned_at": self._now(),
                },
            )
        self.c.event(
            name,
            f"assigned to allocation {alloc['id']} (job {alloc['job_id']}): "
            f"{need} GPU, estimate {hours(estimate)} ({basis})",
        )

    def place(self, pending):
        running = [
            a
            for a in self.active().values()
            if a["status"] == "running"
            and a.get("deadline")
            and a.get("heartbeat")
            and self._now() - a["heartbeat"] <= self.config["heartbeat_timeout"]
            and not (Path(a["dir"]) / "drain").exists()
        ]
        order = sorted(pending, key=lambda n: -self.estimate(n)[0])  # longest first
        placed = set()
        for name in order:
            entry = self.c.state["tasks"][name]
            if self.c._reusable_outputs(name):
                self.c._mark_done(
                    name,
                    entry["attempts"][-1] if entry["attempts"] else None,
                    "valid results found before assignment",
                )
                placed.add(name)
                continue
            need = self.ngpus(name)
            estimate, basis, _ = self.estimate(name)
            if need > self.config["gpus"]:
                entry.update(
                    status="held",
                    reason=f"needs {need} GPUs; allocations have {self.config['gpus']}",
                )
                self.c.event(name, entry["reason"], "error")
                placed.add(name)
                continue
            if estimate + self.config["start_margin"] > self.config["time_seconds"]:
                entry.update(
                    status="held",
                    reason=f"estimate {hours(estimate)} exceeds the allocation time {self.config['time']}",
                )
                self.c.event(name, entry["reason"], "error")
                placed.add(name)
                continue
            for alloc in sorted(running, key=lambda a: a["deadline"]):
                free = alloc["gpus"] - self.assigned_gpus(alloc)
                remaining = alloc["deadline"] - self._now()
                if need <= free and estimate + self.config["start_margin"] <= remaining:
                    self.assign(name, alloc, need, estimate, basis)
                    placed.add(name)
                    break
        return [n for n in pending if n not in placed]

    # ---- demand ----------------------------------------------------------

    def balance(self, waiting):
        active = self.active()
        if not waiting:
            for alloc in active.values():
                if (
                    alloc["status"] == "queued"
                    and not alloc["assigned"]
                    and alloc.get("job_id")
                ):
                    alloc["status_note"] = "not needed"
                    try:
                        self.c.slurm.cancel(alloc["job_id"], pending_only=True)
                        self.c.event(
                            None,
                            f"allocation {alloc['id']} no longer needed; cancelling it while pending",
                        )
                    except Exception as exc:  # noqa: BLE001
                        self.c.event(
                            None,
                            f"could not cancel allocation {alloc['id']}: {exc}",
                            "warning",
                        )
                elif alloc["status"] == "running" and not alloc["assigned"]:
                    flag = Path(alloc["dir"]) / "drain"
                    if not flag.exists():
                        if not self.c.dry_run:
                            flag.write_text("no work left\n")
                        self.c.event(
                            None,
                            f"allocation {alloc['id']} has no work left; asked it to exit",
                        )
            return
        for alloc in active.values():
            flag = Path(alloc["dir"]) / "drain"
            if flag.exists() and not self.c.dry_run:
                flag.unlink()
        if self._now() < self.c.state.get("pack_blocked_until", 0):
            return
        demand = sum(self.ngpus(n) for n in waiting)
        spare = sum(
            a["gpus"] - self.assigned_gpus(a)
            for a in active.values()
            if a["status"] != "running"
        )
        if len(active) < self.config["max_allocations"] and demand > spare:
            self.submit_allocation(
                f"{len(waiting)} task(s) waiting, {demand} GPU(s) needed"
            )

    def step(self, queue):
        self.update(queue)
        pending = self.assignable()
        waiting = self.place(pending)
        self.balance(waiting)

    def cancel_task(self, name, then):
        entry = self.c.state["tasks"][name]
        attempt = entry["attempts"][-1]
        alloc = self.allocs.get(attempt.get("allocation"))
        attempt["cancel_intent"] = then
        if alloc is None or name not in alloc.get("assigned", {}):
            return False
        if not self.c.dry_run:
            fp_pilot.atomic_json(
                Path(alloc["dir"]) / "cancel" / f"{name}.json", {"intent": then}
            )
        self.c.event(name, f"asked allocation {alloc['id']} to stop it ({then})")
        return True


def summary_rows(state):
    rows = []
    for alloc_id, alloc in sorted(state.get("allocations", {}).items()):
        remaining = (alloc.get("deadline") or 0) - time.time()
        rows.append(
            {
                "allocation": alloc_id,
                "job_id": alloc.get("job_id") or "-",
                "status": alloc["status"],
                "node": alloc.get("node") or "-",
                "gpus": f"{sum(i['ngpus'] for i in alloc.get('assigned', {}).values())}/{alloc['gpus']} busy",
                "time_left": hours(remaining)
                if alloc["status"] == "running" and remaining > 0
                else "-",
                "tasks": ", ".join(sorted(alloc.get("assigned", {}))) or "-",
            }
        )
    return rows
