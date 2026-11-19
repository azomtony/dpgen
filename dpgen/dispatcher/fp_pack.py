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
import re
import shutil
import statistics
import time
from pathlib import Path

from dpgen.dispatcher import fp_recovery_worker as worker
from dpgen.dispatcher import fp_vasp_monitor as vasp_monitor
from dpgen.dispatcher import node_pool

ALLOCATIONS = "allocations"
seconds = node_pool.seconds
hours = node_pool.hours
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
    "shared": False,
}


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
    for key in ("enabled", "shared"):
        if not isinstance(config[key], bool):
            raise ValueError(f"fp.slurm.pack.{key} must be boolean")
    if not config["enabled"]:
        # Still usable to finish allocations started before pack was switched off.
        for key in ("time", "default_runtime"):
            try:
                config[f"{key}_seconds"] = seconds(config[key])
            except ValueError:
                config[f"{key}_seconds"] = seconds(PACK_DEFAULTS[key])
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


def stage_pool_config(campaign, pack):
    """Pool settings for allocations that belong to one FP stage."""
    return node_pool.pool_options(
        {
            "gpus": pack["gpus"],
            "time": pack["time"],
            "max_allocations": pack["max_allocations"],
            "directives": pack["directives"],
            "prepend": campaign.config["prepend"],
            "vars": dict(campaign.config["vars"], iter=campaign.iter_label),
            "python": campaign.config["python"],
            "idle_exit": pack["idle_exit"],
            "start_margin": pack["start_margin"],
            "heartbeat_timeout": pack["heartbeat_timeout"],
            "warn_before_end": pack["warn_before_end"],
            "job_name": f"{campaign.iter_label}-fp-pack-{{allocation}}",
            "keep": 0,  # a stage pool keeps nothing once its work is done
            "accounting_grace": campaign.config["accounting_grace"],
        },
        label="fp.slurm.pack",
    )


class PackManager:
    """Assigns one FP stage's tasks to the allocations of a node pool.

    The pool is the stage's own (``02.fp/allocations``) or, with
    ``pack.shared``, the run-wide pool that training and exploration use too.
    """

    def __init__(self, campaign, config):
        self.c = campaign
        self.config = config
        self.shared = bool(config.get("shared"))
        self.owner = f"fp:{campaign.iter_label}"
        if self.shared:
            root = node_pool.pool_root(campaign.root.parent.parent)
            pool_config = campaign.config.get("pool") or node_pool.load_config(root)
        else:
            root = campaign.root / ALLOCATIONS
            pool_config = stage_pool_config(campaign, config)
        self.pool = node_pool.NodePool(
            root,
            pool_config,
            campaign.slurm,
            clock=campaign.clock,
            event=lambda message, level="info": campaign.event(None, message, level),
            dry_run=campaign.dry_run,
            workdir=campaign.root.parent.parent,
        )
        self.pool.scripts = campaign.alloc_scripts
        self.limits = pool_config

    # ---- helpers ---------------------------------------------------------

    @property
    def allocs(self):
        return self.pool.allocs

    def _now(self):
        return self.c.clock()

    def key(self, name):
        return f"{self.c.iter_label}.{name}"

    def mine(self, alloc):
        return {
            key: item
            for key, item in alloc.get("assigned", {}).items()
            if item.get("owner") == self.owner
        }

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
        ceiling = self.limits["time_seconds"] - self.limits["start_margin"]
        if not confident and value > ceiling:
            value, basis = ceiling, basis + "; uncertain, capped to one allocation"
        floor = entry.get("min_runtime")
        if floor and floor * 1.2 > value:
            return floor * 1.2, "at least what an interrupted attempt already ran", True
        return value, basis, confident

    def active(self):
        return self.pool.active()

    def assigned_gpus(self, alloc):
        return self.pool.used_gpus(alloc)

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

    def update(self, queue):
        self.pool.update(queue)
        for alloc in self.pool.allocs.values():
            if not self.mine(alloc):
                continue
            self.collect(alloc)
            if alloc["status"] not in node_pool.ACTIVE:
                self.end(alloc)

    def collect(self, alloc):
        tasks = self.c.state["tasks"]
        for key, item in list(self.mine(alloc).items()):
            name = item.get("task")
            entry = tasks.get(name)
            if entry is None:
                continue
            record = self.pool.read(alloc, "running", key)
            attempt = entry["attempts"][-1]
            if record and attempt.get("started_at") is None:
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
            record = self.pool.read(alloc, "finished", key)
            if record is None:
                continue
            self.pool.release(alloc, key, suffix=f".a{item.get('attempt')}")
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
        """The allocation left the queue: settle this stage's tasks still on it."""
        for key, item in list(self.mine(alloc).items()):
            name = item.get("task")
            alloc["assigned"].pop(key)
            entry = self.c.state["tasks"].get(name)
            if entry is None:
                continue
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
            self.finish_task(name, alloc, record)

    # ---- assignment ------------------------------------------------------

    def assign(self, name, alloc, need, estimate, basis):
        entry = self.c.state["tasks"][name]
        task = self.c.root / name
        number = (entry["attempts"][-1]["n"] if entry["attempts"] else 0) + 1
        archived = self.c._archive(name)
        _, command, values, script = self.c.render(name, number)
        # Same environment and worker as fp.sbatch, without the #SBATCH lines.
        lines = [line for line in script.splitlines() if not line.startswith("#SBATCH")]
        for index in range(len(lines) - 1, -1, -1):
            if lines[index].endswith("fp_recovery_worker.py"):
                lines[index] = "exec " + lines[index]
                break
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
            (task / "fp-pilot.sh").write_text("\n".join(lines) + "\n")
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
        self.pool.assign(
            alloc,
            self.key(name),
            cwd=task,
            argv=["bash", "fp-pilot.sh"],
            ngpus=need,
            estimate=estimate,
            owner=self.owner,
            task=name,
            attempt=number,
        )
        self.c.event(
            name,
            f"assigned to allocation {alloc['id']} (job {alloc['job_id']}): "
            f"{need} GPU, estimate {hours(estimate)} ({basis})",
        )

    def place(self, pending):
        running = self.pool.running()
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
            if need > self.limits["gpus"]:
                entry.update(
                    status="held",
                    reason=f"needs {need} GPUs; allocations have {self.limits['gpus']}",
                )
                self.c.event(name, entry["reason"], "error")
                placed.add(name)
                continue
            if estimate + self.limits["start_margin"] > self.limits["time_seconds"]:
                entry.update(
                    status="held",
                    reason=f"estimate {hours(estimate)} exceeds the allocation time {self.limits['time']}",
                )
                self.c.event(name, entry["reason"], "error")
                placed.add(name)
                continue
            for alloc in running:
                if self.pool.fits(alloc, need, estimate):
                    self.assign(name, alloc, need, estimate, basis)
                    placed.add(name)
                    break
        return [n for n in pending if n not in placed]

    # ---- demand ----------------------------------------------------------

    def balance(self, waiting):
        shortest = min((self.estimate(n)[0] for n in waiting), default=0)
        self.pool.balance(
            sum(self.ngpus(n) for n in waiting), len(waiting), "fp", shortest
        )

    def step(self, queue):
        self.update(queue)
        pending = self.assignable()
        waiting = self.place(pending)
        self.balance(waiting)
        self.pool.save()

    def cancel_task(self, name, then):
        entry = self.c.state["tasks"][name]
        attempt = entry["attempts"][-1]
        alloc = self.allocs.get(attempt.get("allocation"))
        attempt["cancel_intent"] = then
        if alloc is None or self.key(name) not in alloc.get("assigned", {}):
            return False
        self.pool.cancel(alloc, self.key(name), then)
        self.c.event(name, f"asked allocation {alloc['id']} to stop it ({then})")
        return True


def stage_pools(stage_root):
    """Pool states on disk that may hold this FP stage's allocations."""
    stage_root = Path(stage_root)
    found = []
    for root in (
        stage_root / ALLOCATIONS,
        node_pool.pool_root(stage_root.parent.parent),
    ):
        state = node_pool.read_state(root)
        if state is not None:
            found.append((root, state))
    return found


def summary_rows(allocs, now=None):
    return node_pool.summary_rows(allocs, now)
