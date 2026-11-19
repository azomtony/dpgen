"""In-place Slurm controller for VASP single points (the FP stage).

The controller runs on a cluster login node that shares its filesystem with the
compute nodes. Every job runs directly inside ``iter.XXXXXX/02.fp/task.*``:
there is no staging copy and no hash-named directory. Per task, the directory
holds a readable ``fp.sbatch``, ``slurm-<jobid>.out``, and an ``attempts/NNN``
archive of every earlier attempt. The stage directory holds:

``fp-campaign.json``
    Authoritative per-task state (status, attempts, job IDs, resources).
``fp-status.tsv``
    A human-readable table, rewritten every check.
``fp-events.log``
    One line per submission, failure, escalation and user action.
``fp-requests.json``
    Queue written by ``dpgen fp set|hold|release|resubmit|discard``.

Resources are resolved for each submission from ``machine.json`` (``fp.slurm``):
base ``vars`` → size/system ``rules`` → user overrides → automatic escalation
(for example ``ngpus`` 1 → 2 → 4 after GPU out-of-memory). Changing them never
invalidates the campaign; only the scientific inputs identify a task.

Completed data is never lost: a task whose outputs validate is frozen
(read-only, checksummed in ``fp-done.json``) and is never resubmitted; outputs of
an unsuccessful attempt are moved into ``attempts/NNN`` before a retry; and no
job is submitted into a directory that already has a queued or running job.
"""

import argparse
import fcntl
import getpass
import hashlib
import json
import math
import os
import re
import shutil
import string
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack, contextmanager
from fnmatch import fnmatch
from pathlib import Path

from dpgen import dlog
from dpgen.dispatcher import fp_pack
from dpgen.dispatcher import fp_recovery_worker as worker
from dpgen.dispatcher import fp_vasp_monitor as vasp_monitor

STATE = "fp-campaign.json"
REQUESTS = "fp-requests.json"
EVENTS = "fp-events.log"
STATUS = "fp-status.tsv"
DONE = "fp-done.json"
LEGACY_LEDGER = ".fp-recovery.json"

INPUT_FILES = ("POSCAR", "INCAR", "POTCAR", "KPOINTS")
OUTPUT_FILES = (
    "OUTCAR",
    "OSZICAR",
    "vasprun.xml",
    "CONTCAR",
    "XDATCAR",
    "EIGENVAL",
    "DOSCAR",
    "IBZKPT",
    "PCDAT",
    "REPORT",
    "CHG",
    "CHGCAR",
    "WAVECAR",
    "vaspout.h5",
    "fp.log",
    "STOPCAR",
    "fp-attempt.json",
    "fp-runtime.log",
    "fp-monitor-stop.json",
)
PROTECTED_OUTPUTS = ("OUTCAR", "vasprun.xml", "OSZICAR")
# Not copied out of an old staging folder: not labels, often many GB, and the
# originals remain there.
HARVEST_SKIP = ("STOPCAR", "WAVECAR", "CHG", "CHGCAR", "vaspout.h5")
TERMINAL = ("done", "stopped", "failed", "discarded")
ACTIVE = ("queued", "running")
JOB_ID = re.compile(r"[0-9]+(?:_[0-9]+)?")

DEFAULT_RETRY = {
    "hardware": -1,
    "gpu_initialization": -1,
    "preemption": -1,
    "unknown": 1,
    "timeout": 0,
    "resource": 0,
    "application": 0,
    "configuration": 0,
    "submit_rejected": 0,
}
DEFAULTS = {
    "enabled": False,
    "check_interval": 30,
    "retry_delay": 60,
    "accounting_grace": 180,
    "python": None,  # default: the interpreter running dpgen (shared filesystem)
    "job_name": "{iter}-{task}-{formula}-{natoms}at",
    "directives": [],
    "prepend": [],
    "command": None,
    "vars": {},
    "rules": [],
    "escalate": {},
    "retry": {},
    "on_exhausted": "hold",
    "protect_outputs": True,
    "update_pending": True,
    "site": "default",  # name used for runtime history (one per cluster later)
    "pack": {},
}
META_KEYS = ("iter", "task", "sys", "natoms", "formula", "attempt")


# --------------------------------------------------------------------------
# configuration


def options(value):
    """Validate the ``fp.slurm`` section of machine.json."""
    value = {} if value is None else value
    if not isinstance(value, dict) or value.keys() - DEFAULTS.keys():
        unknown = sorted(set(value) - set(DEFAULTS)) if isinstance(value, dict) else []
        raise ValueError(f"Invalid fp.slurm options: {', '.join(unknown) or value}")
    config = dict(DEFAULTS, **value)
    if not isinstance(config["enabled"], bool):
        raise ValueError("fp.slurm.enabled must be boolean")
    if not config["enabled"]:
        return config
    for key in ("check_interval", "retry_delay", "accounting_grace"):
        number = config[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError(f"fp.slurm.{key} must be a number")
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"fp.slurm.{key} must be finite and non-negative")
    if config["check_interval"] <= 0:
        raise ValueError("fp.slurm.check_interval must be positive")
    if not isinstance(config["command"], str) or not config["command"].strip():
        raise ValueError("fp.slurm.command (the VASP launch command) is required")
    for key in ("directives", "prepend"):
        if not isinstance(config[key], list) or not all(
            isinstance(item, str) for item in config[key]
        ):
            raise ValueError(f"fp.slurm.{key} must be a list of strings")
    for directive in config["directives"]:
        if not directive.startswith("-"):
            raise ValueError(
                f"fp.slurm.directives entries are sbatch options such as "
                f"'--gres=gpu:{{ngpus}}', not {directive!r}"
            )
        if re.match(r"--(?:chdir|output|job-name)\b|-[DoJ]\b", directive):
            raise ValueError(
                f"{directive!r} is set by the controller; remove it from fp.slurm.directives"
            )
    if not isinstance(config["vars"], dict):
        raise ValueError("fp.slurm.vars must be an object")
    _check_var_names(config["vars"], "fp.slurm.vars")
    if not isinstance(config["rules"], list):
        raise ValueError("fp.slurm.rules must be a list")
    for index, rule in enumerate(config["rules"]):
        allowed = {"min_natoms", "max_natoms", "systems", "tasks", "vars"}
        if not isinstance(rule, dict) or rule.keys() - allowed or "vars" not in rule:
            raise ValueError(
                f"fp.slurm.rules[{index}] needs 'vars' plus any of "
                "min_natoms, max_natoms, systems, tasks"
            )
        _check_var_names(rule["vars"], f"fp.slurm.rules[{index}].vars")
    if not isinstance(config["escalate"], dict):
        raise ValueError("fp.slurm.escalate must be an object")
    for category, ladder in config["escalate"].items():
        if not isinstance(ladder, dict) or not all(
            isinstance(values, list) and values for values in ladder.values()
        ):
            raise ValueError(
                f"fp.slurm.escalate.{category} must map variables to non-empty lists"
            )
        _check_var_names(ladder, f"fp.slurm.escalate.{category}")
    retry = dict(DEFAULT_RETRY)
    if not isinstance(config["retry"], dict):
        raise ValueError("fp.slurm.retry must be an object")
    for category, limit in config["retry"].items():
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < -1:
            raise ValueError(
                f"fp.slurm.retry.{category} must be an integer >= -1 (-1 = unlimited)"
            )
        retry[category] = limit
    config["retry"] = retry
    if config["on_exhausted"] not in ("hold", "fail"):
        raise ValueError("fp.slurm.on_exhausted must be 'hold' or 'fail'")
    if not isinstance(config["site"], str) or not config["site"]:
        raise ValueError("fp.slurm.site must be a non-empty string")
    config["pack"] = fp_pack.pack_options(config["pack"], config["vars"])
    for key in ("protect_outputs", "update_pending"):
        if not isinstance(config[key], bool):
            raise ValueError(f"fp.slurm.{key} must be boolean")
    # Every template must render for every task, so placeholders and anything a
    # rule or escalation changes must have a base value in fp.slurm.vars.
    names = set(config["vars"]) | set(META_KEYS)
    for label, values in [
        *((f"rules[{i}]", rule["vars"]) for i, rule in enumerate(config["rules"])),
        *((f"escalate.{c}", ladder) for c, ladder in config["escalate"].items()),
    ]:
        undefined = set(values) - set(config["vars"])
        if undefined:
            raise ValueError(
                f"fp.slurm.{label} changes {sorted(undefined)}, which have no base "
                "value in fp.slurm.vars"
            )
    for label, template in (
        ("job_name", config["job_name"]),
        ("command", config["command"]),
        *((f"directives[{i}]", d) for i, d in enumerate(config["directives"])),
        *((f"prepend[{i}]", p) for i, p in enumerate(config["prepend"])),
    ):
        missing = _placeholders(template) - names
        if missing:
            raise ValueError(
                f"fp.slurm.{label} uses undefined variable(s) {sorted(missing)}; "
                "define them in fp.slurm.vars"
            )
    return config


def _check_var_names(values, label):
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be an object")
    for name in values:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in META_KEYS:
            raise ValueError(f"{label}: invalid or reserved variable name {name!r}")


def _placeholders(template):
    names = set()
    for _, field, _, _ in string.Formatter().parse(template):
        if field is not None:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field):
                raise ValueError(
                    f"Unsupported template field {{{field}}} in {template!r}"
                )
            names.add(field)
    return names


def enabled(mdata):
    value = mdata.get("fp_slurm")
    return isinstance(value, dict) and value.get("enabled", False) is True


def campaign_requested(work_path, mdata):
    """Return True when this FP stage must be run by the in-place controller."""
    exists = (Path(work_path) / STATE).exists()
    if exists and not enabled(mdata):
        raise RuntimeError(
            f"{Path(work_path) / STATE} exists but fp.slurm is not enabled in "
            "machine.json. Re-enable it; falling back to another submitter could "
            "duplicate running jobs."
        )
    return exists or enabled(mdata)


# --------------------------------------------------------------------------
# Slurm commands


class SlurmError(RuntimeError):
    pass


class SubmitRejected(SlurmError):
    """sbatch answered and refused the job; nothing was queued."""


class SubmitUncertain(SlurmError):
    """sbatch did not answer clearly; a job may or may not exist."""


class Slurm:
    """Thin wrapper over the Slurm CLI on the login node."""

    def __init__(self, run=subprocess.run, user=None):
        self.run = run
        self.user = user or getpass.getuser()

    def _call(self, args, cwd=None, timeout=120):
        try:
            return self.run(
                args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SlurmError(
                f"{args[0]} not found; run dpgen on the cluster login node"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SlurmError(f"{args[0]} timed out after {timeout} s") from exc

    def submit(self, script, cwd):
        try:
            result = self._call(["sbatch", "--parsable", script], cwd=cwd, timeout=300)
        except (OSError, SlurmError) as exc:
            raise SubmitUncertain(f"sbatch did not complete: {exc}") from exc
        output = (result.stdout or "").strip()
        error = (result.stderr or "").strip()
        if result.returncode == 0:
            job_id = output.splitlines()[-1].split(";")[0].strip() if output else ""
            if JOB_ID.fullmatch(job_id):
                return job_id
            raise SubmitUncertain(f"sbatch succeeded without a job id: {output!r}")
        if re.search(r"time(?:d)?\s*out|socket|connection|try again", error, re.I):
            raise SubmitUncertain(f"sbatch: {error}")
        raise SubmitRejected(error or output or f"sbatch exit {result.returncode}")

    def cluster(self):
        """Slurm ClusterName of this login node, or None if it cannot be read."""
        try:
            result = self._call(["scontrol", "show", "config"], timeout=60)
        except SlurmError:
            return None
        match = re.search(r"^\s*ClusterName\s*=\s*(\S+)", result.stdout or "", re.M)
        return match[1] if match else None

    def queue(self):
        """Jobs of this user still known to the controller: id -> state/workdir/name."""
        result = self._call(["squeue", "-h", "-u", self.user, "-o", "%i|%T|%Z|%j"])
        if result.returncode:
            raise SlurmError(f"squeue failed: {(result.stderr or '').strip()}")
        jobs = {}
        for line in (result.stdout or "").splitlines():
            parts = line.strip().split("|", 3)
            if len(parts) == 4 and parts[0]:
                jobs[parts[0]] = {
                    "state": parts[1],
                    "workdir": parts[2],
                    "name": parts[3],
                }
        return jobs

    def accounting(self, job_ids):
        job_ids = [job_id for job_id in job_ids if JOB_ID.fullmatch(str(job_id))]
        if not job_ids:
            return {}
        result = self._call(
            [
                "sacct",
                "-n",
                "-X",
                "-P",
                "-j",
                ",".join(job_ids),
                "--format=JobIDRaw,State,ExitCode,NodeList,Partition,Elapsed",
            ]
        )
        if result.returncode:
            dlog.warning("sacct failed: %s", (result.stderr or "").strip())
            return {}
        rows = {}
        for line in (result.stdout or "").splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 6:
                rows[parts[0]] = dict(
                    zip(
                        (
                            "job_id",
                            "state",
                            "exit_code",
                            "node",
                            "partition",
                            "elapsed",
                        ),
                        parts[:6],
                    )
                )
        return rows

    def find(self, workdir, since):
        """Job IDs submitted from ``workdir`` since ``since`` (epoch seconds)."""
        start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(since))
        result = self._call(
            [
                "sacct",
                "-n",
                "-X",
                "-P",
                "-u",
                self.user,
                "-S",
                start,
                "--format=JobIDRaw,WorkDir",
            ]
        )
        if result.returncode:
            raise SlurmError(f"sacct failed: {(result.stderr or '').strip()}")
        found = []
        for line in (result.stdout or "").splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 2 and _same_path(parts[1], workdir):
                found.append(parts[0])
        return found

    def cancel(self, job_id, pending_only=False):
        """Cancel a job; with ``pending_only`` Slurm skips it once it has started."""
        if not JOB_ID.fullmatch(str(job_id)):
            raise SlurmError(f"Refusing to cancel invalid job id {job_id!r}")
        state = ["--state=PENDING"] if pending_only else []
        result = self._call(["scancel", *state, str(job_id)])
        if result.returncode:
            raise SlurmError(
                f"scancel {job_id} failed: {(result.stderr or '').strip()}"
            )


def _same_path(left, right):
    try:
        return os.path.realpath(str(left)) == os.path.realpath(str(right))
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------------
# helpers


def poscar_info(path):
    """Return (natoms, formula) from a VASP 5 POSCAR."""
    lines = Path(path).read_text().splitlines()
    if len(lines) < 7:
        raise ValueError(f"Malformed POSCAR: {path}")
    species, counts = lines[5].split(), lines[6].split()
    if not all(count.isdigit() for count in counts):
        # VASP 4 file without a species line.
        counts, species = species, [f"X{i}" for i in range(len(species))]
    counts = [int(count) for count in counts]
    formula = "".join(f"{s}{n}" for s, n in zip(species, counts) if n)
    return sum(counts), formula


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _magnitude(value):
    """Comparable size for a resource value (count, wall time or memory)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGT])i?B?", text, re.I)
    if match:
        power = "KMGT".index(match[2].upper()) + 1
        return float(match[1]) * 1024**power
    # Slurm wall time: M:S, H:M:S, D-H, D-H:M, D-H:M:S.
    match = re.fullmatch(r"(?:(\d+)-)?(\d+(?::\d+){0,2})", text)
    if match and (match[1] or ":" in match[2]):
        parts = [int(p) for p in match[2].split(":")]
        if match[1]:
            hours, minutes, seconds = (parts + [0, 0])[:3]
        elif len(parts) == 2:
            hours, (minutes, seconds) = 0, parts
        else:
            hours, minutes, seconds = parts
        return float(int(match[1] or 0) * 86400 + hours * 3600 + minutes * 60 + seconds)
    return None


def next_rung(current, ladder):
    """The first ladder value strictly larger than ``current``."""
    size = _magnitude(current)
    if size is not None:
        for value in ladder:
            other = _magnitude(value)
            if other is not None and other > size:
                return value
        return None
    if current in ladder:
        index = ladder.index(current)
        return ladder[index + 1] if index + 1 < len(ladder) else None
    return ladder[0]


def _now_text(epoch=None):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


@contextmanager
def _flock(path, blocking=True):
    with open(path, "a+") as handle:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle, flags)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another dpgen controller is already managing {Path(path).parent}"
            ) from exc
        yield


def read_requests(root):
    path = Path(root) / REQUESTS
    if not path.exists():
        return {"version": 1, "requests": []}
    return json.loads(path.read_text())


def _cluster_of(slurm):
    getter = getattr(slurm, "cluster", None)
    return getter() if callable(getter) else None


class DryRunSlurm:
    """Reads the real queue and accounting; records submissions and cancels."""

    def __init__(self, real):
        self.real = real
        self.submitted = []
        self.cancelled = []

    def queue(self):
        return self.real.queue()

    def cluster(self):
        return _cluster_of(self.real)

    def accounting(self, job_ids):
        return self.real.accounting(job_ids)

    def find(self, workdir, since):
        return self.real.find(workdir, since)

    def submit(self, script, cwd):
        self.submitted.append(cwd)
        return f"NEW{len(self.submitted)}"

    def cancel(self, job_id, pending_only=False):
        self.cancelled.append((str(job_id), pending_only))


# --------------------------------------------------------------------------
# controller


class Campaign:
    def __init__(
        self,
        root,
        config,
        jdata=None,
        slurm=None,
        clock=time.time,
        sleep=time.sleep,
        validator=None,
        dry_run=False,
        moved=False,
    ):
        from dpgen.dispatcher.fp_recovery import validate_vasp_outputs

        self.moved = moved  # stage was copied here from another cluster
        self.root = Path(root).resolve()
        self.config = config
        self.jdata = jdata or {}
        self.slurm = slurm or Slurm()
        self.clock = clock
        self.sleep = sleep
        self.validate = validator or validate_vasp_outputs
        self.monitor = vasp_monitor.options(self.jdata.get("fp_monitor"))
        self.health_path = self.root.parent.parent / ".dpgen-gpu-health.json"
        self.iter_label = self._iter_label()
        self.state = None
        # Dry run: nothing is submitted, cancelled, moved, copied or written.
        self.dry_run = dry_run
        self.plan = []  # (task label, message, level)
        self.scripts = {}  # task -> fp.sbatch text that would be written
        self.view = {}  # task -> directory read in place of the task (dry run)
        if dry_run:
            self.slurm = DryRunSlurm(self.slurm)
        self.alloc_scripts = {}  # allocation id -> alloc.sbatch text (dry run)
        self.pack = None
        pack = config.get("pack") or {}
        self.runtime = fp_pack.RuntimeModel(
            self.root.parent.parent / fp_pack.RUNTIME_HISTORY,
            pack.get("default_runtime_seconds", 7200),
            config.get("site", "default"),
        )

    @staticmethod
    def _quote(text):
        return _shell_quote(text)

    @staticmethod
    def _same_path(left, right):
        return _same_path(left, right)

    # ---- persistence -----------------------------------------------------

    def _iter_label(self):
        match = re.fullmatch(r"iter\.(\d+)", self.root.parent.name)
        return f"i{int(match[1]):03d}" if match else self.root.parent.name

    def save(self):
        if self.dry_run:
            return
        self.state["updated"] = self.clock()
        worker.atomic_json(self.root / STATE, self.state)
        self.write_status()

    def event(self, name, message, level="info"):
        entry = self.state["tasks"].get(name, {}) if name else {}
        label = (
            f"{name} [{entry.get('formula')} {entry.get('natoms')}at]"
            if entry
            else "stage"
        )
        if self.dry_run:
            self.plan.append((label, message, level))
            return
        line = f"{_now_text(self.clock())} {label} {message}"
        with (self.root / EVENTS).open("a") as stream:
            stream.write(line + "\n")
        getattr(dlog, level)("FP %s %s: %s", self.iter_label, label, message)

    def write_status(self):
        rows = status_rows(self.state, self.config)
        header = (
            "task",
            "natoms",
            "formula",
            "status",
            "attempt",
            "job_id",
            "resources",
            "reason",
        )
        lines = ["\t".join(header)]
        lines += ["\t".join(str(row[key]) for key in header) for row in rows]
        temporary = self.root / (STATUS + ".tmp")
        temporary.write_text("\n".join(lines) + "\n")
        os.replace(temporary, self.root / STATUS)

    # ---- task setup ------------------------------------------------------

    def _task_dirs(self):
        return sorted(path.name for path in self.root.glob("task.*") if path.is_dir())

    def _new_entry(self, name):
        natoms, formula = poscar_info(self.root / name / "POSCAR")
        match = re.fullmatch(r"task\.(\d+)\.(\d+)", name)
        return {
            "status": "pending",
            "natoms": natoms,
            "formula": formula,
            "sys": match[1] if match else "",
            "input_digest": worker.input_digest(self.root / name),
            "manual": {},
            "escalated": {},
            "attempts": [],
            "retries": {},
            "reason": "",
        }

    def initialize(self, queue):
        path = self.root / STATE
        if path.exists():
            self.state = json.loads(path.read_text())
            self._check_cluster()
            self._reconcile_inputs()
            return
        self.state = {"version": 1, "created": self.clock(), "tasks": {}}
        legacy = self.root / LEGACY_LEDGER
        if legacy.exists():
            self._adopt_legacy(json.loads(legacy.read_text()), queue)
        else:
            self._refuse_untracked_jobs(queue, [self.root])
            for name in self._task_dirs():
                self.state["tasks"][name] = entry = self._new_entry(name)
                if self._has_valid_outputs(name):
                    self._mark_done(
                        name, None, "valid results present before submission"
                    )
        here = _cluster_of(self.slurm)
        if here:
            self.state["cluster"] = here
        self.save()
        self.event(None, f"campaign initialized with {len(self.state['tasks'])} tasks")

    # ---- moving a stage between clusters ----------------------------------

    def _tracks_jobs(self):
        tasks = self.state["tasks"].values()
        allocations = [
            alloc
            for _, pool in fp_pack.stage_pools(self.root)
            for alloc in pool["allocations"].values()
        ]
        return any(e["status"] in ACTIVE or e.get("submitting") for e in tasks) or any(
            a.get("status") in ("submitting", "queued", "running") for a in allocations
        )

    def _check_cluster(self):
        """Refuse to treat another cluster's job IDs as ours; adopt a moved stage."""
        here = _cluster_of(self.slurm)
        before = self.state.get("cluster")
        if self.moved:
            self._move_here(before or "the previous cluster", here or "this cluster")
        elif before and here and before != here:
            if self._tracks_jobs():
                raise RuntimeError(
                    f"This FP stage was last run on cluster {before!r}, and this is "
                    f"{here!r}. Its queued/running jobs belong to {before}. If you "
                    "copied the stage here to finish it, run once with --moved "
                    "(`dpgen fp run PARAM MACHINE --moved`). Nothing was submitted."
                )
            self.event(
                None, f"stage moved from {before} to {here}; no jobs were active"
            )
        if here:
            self.state["cluster"] = here

    def _move_here(self, before, here):
        """Forget the other cluster's jobs; keep every result that validates.

        A task whose job finished there before the copy becomes done; any other
        unfinished task is submitted again here (its partial outputs are archived
        first, as for any retry). Held, failed, stopped and discarded tasks keep
        their status. Moving never uses up a retry.
        """
        now = self.clock()
        resubmit, finished = [], []
        for name, entry in self.state["tasks"].items():
            if entry["status"] in TERMINAL:
                continue
            if entry["status"] not in ACTIVE and not entry.get("submitting"):
                continue
            attempt = entry["attempts"][-1] if entry["attempts"] else None
            if attempt is not None and "ended" not in attempt:
                attempt.update(
                    category="moved",
                    reason=f"stage moved from {before} to {here}",
                    ended=now,
                )
            for key in ("submitting", "left_queue_at", "foreign_warned"):
                entry.pop(key, None)
            if self._reusable_outputs(name):
                self._mark_done(name, attempt, f"finished on {before} before the move")
                finished.append(name)
                continue
            entry.update(
                status="pending",
                retry_after=0,
                reason=f"moved from {before}; job there no longer tracked",
            )
            resubmit.append(name)
        for pool_dir, pool in fp_pack.stage_pools(self.root):
            changed = False
            for alloc in pool["allocations"].values():
                if alloc.get("status") in ("submitting", "queued", "running"):
                    alloc.update(
                        status="ended", ended=now, reason=f"moved from {before}"
                    )
                    changed = True
            if changed and not self.dry_run:
                worker.atomic_json(pool_dir / fp_pack.node_pool.STATE, pool)
        jobs_left = len(resubmit)
        self.event(
            None,
            f"stage moved from {before} to {here}: {len(finished)} task(s) finished "
            f"there before the copy, {jobs_left} unfinished task(s) will run here. "
            f"Cancel the remaining jobs of this stage on {before}.",
            "warning" if jobs_left else "info",
        )
        self.state.setdefault("moves", []).append(
            {
                "from": before,
                "to": here,
                "time": now,
                "finished": finished,
                "resubmit": resubmit,
            }
        )
        self.moved = False

    def _refuse_untracked_jobs(self, queue, roots):
        tracked = {
            str(attempt.get("job_id"))
            for entry in (self.state or {}).get("tasks", {}).values()
            for attempt in entry.get("attempts", [])
        }
        tracked |= {
            str(alloc.get("job_id"))
            for _, pool in fp_pack.stage_pools(self.root)
            for alloc in pool["allocations"].values()
        }
        stray = sorted(
            job_id
            for job_id, job in queue.items()
            if job_id not in tracked
            and (
                any(
                    _same_path(job["workdir"], root)
                    or os.path.realpath(job["workdir"]).startswith(
                        os.path.realpath(root) + os.sep
                    )
                    for root in roots
                )
                or self._stages_this_stage(job["workdir"])
            )
        )
        if stray:
            raise RuntimeError(
                "Jobs from an earlier submission are still queued or running in "
                f"{self.root}: {', '.join(stray)}. Wait for them, cancel them, or "
                "restore the .fp-recovery.json that tracks them so they can be adopted. "
                "Nothing was submitted."
            )

    def _stages_this_stage(self, workdir):
        """True if ``workdir`` is a dispatcher staging copy of one of our tasks.

        Plain DPDispatcher submissions run in ``<remote_root>/<hash>/task.*``,
        outside 02.fp. Recognize them by an identical task POSCAR so they are
        never duplicated, whether or not an old recovery record exists.
        """
        try:
            staged = Path(workdir)
            if not staged.is_dir() or _same_path(staged, self.root):
                return False
            for path in staged.glob("task.*/POSCAR"):
                ours = self.root / path.parent.name / "POSCAR"
                if ours.is_file() and file_digest(ours) == file_digest(path):
                    return True
        except OSError:
            return False
        return False

    def _reconcile_inputs(self):
        for name in self._task_dirs():
            if name not in self.state["tasks"]:
                self.state["tasks"][name] = self._new_entry(name)
                self.event(name, "new task directory added to the campaign")
        for name, entry in self.state["tasks"].items():
            task = self.root / name
            digest = worker.input_digest(task)
            if entry["status"] == "done":
                self._verify_done(name)
                if digest != entry["input_digest"]:
                    raise RuntimeError(
                        f"Inputs of completed task {name} changed after its result was "
                        "validated. Restore the original POSCAR/INCAR/POTCAR/KPOINTS; "
                        "the validated result is kept and nothing was submitted."
                    )
            elif digest != entry["input_digest"]:
                if entry["status"] in ACTIVE:
                    raise RuntimeError(
                        f"Inputs of {name} changed while job "
                        f"{entry['attempts'][-1].get('job_id')} is active; restore them "
                        "or cancel the job first."
                    )
                entry["input_digest"] = digest
                entry["natoms"], entry["formula"] = poscar_info(task / "POSCAR")
                # Outputs in the folder belong to the old inputs: never label them.
                archived = self._archive(name, label="inputs-changed")
                self.event(
                    name,
                    "inputs changed; next attempt uses the new inputs"
                    + (
                        f" (old outputs kept in {archived.relative_to(task)})"
                        if archived
                        else ""
                    ),
                )

    def _verify_done(self, name):
        record_path = self.root / name / DONE
        if not record_path.exists():
            raise RuntimeError(f"{record_path} is missing for a completed task")
        record = json.loads(record_path.read_text())
        for filename, digest in record.get("outputs", {}).items():
            path = self.root / name / filename
            if not path.exists() or file_digest(path) != digest:
                raise RuntimeError(
                    f"Validated output {path} is missing or modified. Restore it from "
                    "backup; the controller will not overwrite or resubmit a completed task."
                )

    def _has_valid_outputs(self, name):
        task = Path(self.view.get(name, self.root / name))
        return (task / "OUTCAR").exists() and self.validate(task)[0]

    def _reusable_outputs(self, name):
        """Valid results that may become labels (not stopped by the FP monitor)."""
        task = Path(self.view.get(name, self.root / name))
        return self._has_valid_outputs(name) and _stop_reason(task) is None

    def _mark_done(self, name, attempt, reason):
        entry = self.state["tasks"][name]
        task = self.root / name
        outputs = {
            filename: file_digest(task / filename)
            for filename in PROTECTED_OUTPUTS
            if (task / filename).exists()
        }
        record = {
            "task": name,
            "time": self.clock(),
            "job_id": attempt.get("job_id") if attempt else None,
            "attempt": attempt.get("n") if attempt else None,
            "input_digest": entry["input_digest"],
            "outputs": outputs,
            "reason": reason,
        }
        if self.dry_run:
            pass
        else:
            worker.atomic_json(task / DONE, record)
        if self.config.get("protect_outputs", True) and not self.dry_run:
            for filename in outputs:
                path = task / filename
                path.chmod(path.stat().st_mode & ~0o222)
        entry.update(status="done", reason=reason)
        if attempt:
            ngpus = (attempt.get("vars") or {}).get("ngpus", 1)
            try:
                ngpus = int(ngpus)
            except (TypeError, ValueError):
                ngpus = 1
            self.runtime.record(
                entry["natoms"],
                ngpus,
                fp_pack.elapsed_of(task, {"elapsed": attempt.get("elapsed")}),
                name,
                dry_run=self.dry_run,
            )
        entry.pop("retry_after", None)
        self.event(
            name,
            f"done ({reason})"
            + (
                f", job {attempt['job_id']}"
                if attempt and attempt.get("job_id")
                else ""
            ),
        )

    # ---- legacy adoption -------------------------------------------------

    def _adopt_legacy(self, ledger, queue):
        legacy_root = Path(ledger.get("remote_root") or "")
        if ledger.get("remote_root") and not legacy_root.is_dir():
            raise RuntimeError(
                f"The previous FP staging directory {legacy_root} is not visible from "
                "this machine. Run dpgen on the cluster login node that shares this "
                "filesystem so running jobs and finished results can be adopted."
            )
        self._refuse_untracked_jobs(
            {
                job_id: job
                for job_id, job in queue.items()
                if job_id
                not in {str(e.get("job_id")) for e in ledger.get("tasks", {}).values()}
            },
            [self.root] + ([legacy_root] if ledger.get("remote_root") else []),
        )
        self.state["adopted_from"] = {
            "ledger": str(self.root / LEGACY_LEDGER),
            "remote_root": str(legacy_root) if ledger.get("remote_root") else None,
            "submission_hash": ledger.get("submission_hash"),
            "time": self.clock(),
        }
        for name in self._task_dirs():
            self.state["tasks"][name] = entry = self._new_entry(name)
            old = ledger.get("tasks", {}).get(name, {})
            legacy_dir = legacy_root / name if ledger.get("remote_root") else None
            job_id = str(old.get("job_id") or "")
            if self._has_valid_outputs(name):
                self._mark_done(name, None, "validated result from previous controller")
                continue
            if old.get("status") in ("discarded", "discard_pending"):
                entry.update(status="discarded", reason=old.get("reason", "discarded"))
                continue
            if vasp_monitor.stopped_task(self.root / name):
                entry.update(status="stopped", reason="stopped by FP monitor")
                continue
            if job_id and job_id in queue:
                entry["attempts"].append(
                    {
                        "n": 1,
                        "job_id": job_id,
                        "legacy_dir": str(legacy_dir) if legacy_dir else None,
                        "vars": {},
                        "submitted_at": None,
                        "adopted": True,
                    }
                )
                entry["status"] = (
                    "queued" if queue[job_id]["state"] == "PENDING" else "running"
                )
                entry["reason"] = "adopted running job from previous controller"
                self.event(name, f"adopted job {job_id} ({queue[job_id]['state']})")
                continue
            if legacy_dir and (legacy_dir / "OUTCAR").exists():
                attempt = {
                    "n": 1,
                    "job_id": job_id or None,
                    "legacy_dir": str(legacy_dir),
                    "vars": {},
                    "adopted": True,
                }
                entry["attempts"].append(attempt)
                self._harvest_legacy(name, attempt)
                if self._has_valid_outputs(name):
                    self._mark_done(
                        name, attempt, "result copied from previous staging"
                    )
                    continue
                outputs = Path(self.view.get(name, self.root / name))
                stop = _stop_reason(outputs)
                if stop:
                    attempt.update(category="scientific_stop", reason=stop)
                    entry.update(status="stopped", reason=stop)
                    self.event(
                        name,
                        f"stopped by the FP monitor under the previous controller "
                        f"({stop}); excluded, not rerun. `dpgen fp release` reruns it.",
                    )
                    continue
                previous = _previous_failure(outputs, old)
                if previous:
                    attempt.update(reason=previous)
                    entry["reason"] = f"previous attempt: {previous}"
            if old.get("status") == "failed":
                entry.update(
                    status="held",
                    reason="exhausted retries under the previous controller: "
                    + str(old.get("reason", "")),
                )
            note = f" ({entry['reason']})" if entry.get("reason") else ""
            self.event(name, f"adopted as {entry['status']}{note}")

    def _harvest_legacy(self, name, attempt):
        source = Path(attempt["legacy_dir"])
        task = self.root / name
        if not source.is_dir() or _same_path(source, task):
            return  # outputs were written in place already
        if self.dry_run:
            present = [
                f
                for f in OUTPUT_FILES
                if (source / f).is_file() and f not in HARVEST_SKIP
            ]
            self.view[name] = str(source)
            if present:
                self.event(
                    name, f"copied {', '.join(present)} from {source} (left in place)"
                )
            return
        self._archive(name, label="local-before-adopt")
        copied = []
        for filename in OUTPUT_FILES:
            if (source / filename).is_file() and filename not in HARVEST_SKIP:
                shutil.copy2(source / filename, task / filename)
                copied.append(filename)
        attempt["harvested"] = copied
        if copied:
            self.event(
                name, f"copied {', '.join(copied)} from {source} (left in place)"
            )

    # ---- requests --------------------------------------------------------

    def apply_requests(self):
        path = self.root / REQUESTS
        if not path.exists():
            return
        if self.dry_run:
            for request in read_requests(self.root).get("requests", []):
                self._apply(request)
            return
        with _flock(self.root / ".fp-requests.lock"):
            queue = read_requests(self.root)
            pending = queue.get("requests", [])
            if not pending:
                return
            for request in pending:
                self._apply(request)
            log = self.state.setdefault("request_log", [])
            log.extend(dict(r, applied_at=self.clock()) for r in pending)
            queue["requests"] = []
            self.save()
            worker.atomic_json(path, queue)

    def _apply(self, request):
        action = request["action"]
        who = f"request {request['id'][:8]} ({action})"
        for name in request["tasks"]:
            entry = self.state["tasks"].get(name)
            if entry is None:
                self.event(None, f"{who}: unknown task {name} ignored", "warning")
                continue
            if entry["status"] == "done":
                self.event(name, f"{who}: ignored, task already has a validated result")
                continue
            if action == "set":
                entry["manual"].update(request["vars"])
                for key in request["vars"]:
                    entry["escalated"].pop(key, None)
                self.event(
                    name,
                    f"{who}: resources set {_fmt_vars(request['vars'])}"
                    + (
                        " (resubmitting)"
                        if request.get("resubmit")
                        else " (next attempt)"
                    ),
                )
                if request.get("resubmit"):
                    self._restart(name, "resubmit")
            elif action == "resubmit":
                self._restart(name, "resubmit")
                self.event(name, f"{who}: resubmit requested")
            elif action == "hold":
                entry["hold"] = True
                if entry["status"] in ACTIVE and request.get("cancel"):
                    self._cancel(name, "hold")
                elif entry["status"] not in ACTIVE and entry["status"] not in TERMINAL:
                    entry["status"] = "held"
                    entry["reason"] = request.get("reason") or "held by user"
                self.event(name, f"{who}: held")
            elif action == "release":
                entry.pop("hold", None)
                if entry["status"] in ("held", "failed", "stopped"):
                    entry.update(status="pending", retries={}, retry_after=0)
                    entry["reason"] = "released by user"
                self.event(name, f"{who}: released")
            elif action == "discard":
                entry["discard"] = request.get("reason") or "discarded by user"
                if entry["status"] in ACTIVE:
                    self._cancel(name, "discard")
                else:
                    entry.update(status="discarded", reason=entry["discard"])
                self.event(name, f"{who}: discard ({entry['discard']})")
            else:
                self.event(name, f"{who}: unknown action ignored", "warning")

    def _restart(self, name, then):
        entry = self.state["tasks"][name]
        entry.pop("hold", None)
        if entry["status"] in ACTIVE:
            self._cancel(name, then)
        elif entry["status"] != "discarded":
            entry.update(status="pending", retries={}, retry_after=0)

    def _refresh_pending(self, name, attempt):
        """Replace a not-yet-started job whose resources are now out of date."""
        if not self.config["update_pending"] or attempt.get("cancel_intent"):
            return
        wanted = self.resolve_vars(name)
        requested = attempt.get("vars") or {}
        if requested:
            stale = wanted != requested
            before = _fmt_vars(
                {k: requested.get(k) for k in wanted if wanted[k] != requested.get(k)}
            )
        else:
            # Adopted job with an unknown request: act only on tasks that a
            # rule, override or escalation explicitly sends elsewhere.
            stale = wanted != self.config["vars"]
            before = "unknown (adopted job)"
        if not stale:
            return
        job_id = attempt.get("job_id")
        after = _fmt_vars(
            {k: v for k, v in wanted.items() if not requested or requested.get(k) != v}
        )
        attempt["cancel_intent"] = "update"
        try:
            self.slurm.cancel(job_id, pending_only=True)
        except SlurmError as exc:
            attempt.pop("cancel_intent")
            self.event(name, f"could not update pending job {job_id}: {exc}", "error")
            return
        self.event(
            name,
            f"pending job {job_id} requested {before}; task now needs {after}; "
            "replacing it before it starts",
        )

    def _cancel(self, name, then):
        entry = self.state["tasks"][name]
        attempt = entry["attempts"][-1]
        job_id = attempt.get("job_id")
        if attempt.get("allocation"):
            # Never scancel a shared allocation: ask its worker to stop this task.
            self._pack().cancel_task(name, then)
            return
        attempt["cancel_intent"] = then
        try:
            self.slurm.cancel(job_id)
            self.event(name, f"cancelling job {job_id} ({then})")
        except SlurmError as exc:
            self.event(name, f"could not cancel job {job_id}: {exc}", "error")

    # ---- resource resolution ---------------------------------------------

    def resolve_vars(self, name):
        entry = self.state["tasks"][name]
        values = dict(self.config["vars"])
        for rule in self.config["rules"]:
            if _rule_matches(rule, name, entry):
                values.update(rule["vars"])
        values.update(entry.get("manual", {}))
        values.update(entry.get("escalated", {}))
        return values

    def render(self, name, attempt_number):
        entry = self.state["tasks"][name]
        values = self.resolve_vars(name)
        context = dict(
            values,
            iter=self.iter_label,
            task=name.removeprefix("task."),
            sys=entry.get("sys", ""),
            natoms=entry["natoms"],
            formula=entry["formula"],
            attempt=attempt_number,
        )
        job_name = re.sub(
            r"[^A-Za-z0-9._-]", "_", self.config["job_name"].format_map(context)
        )
        command = self.config["command"].format_map(context)
        task = self.root / name
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            "#SBATCH --output=slurm-%j.out",
            f"#SBATCH --chdir={task}",
        ]
        lines += [
            f"#SBATCH {directive.format_map(context)}"
            for directive in self.config["directives"]
        ]
        lines += [
            "",
            f"# dpgen FP {self.iter_label} {name}: {entry['formula']}, "
            f"{entry['natoms']} atoms, attempt {attempt_number}",
            f"# resources: {_fmt_vars(values)}",
            f"# command:   {command}",
            "# Change resources with `dpgen fp set`, not by editing this file.",
            "",
        ]
        lines += [line.format_map(context) for line in self.config["prepend"]]
        lines += [
            f"cd {_shell_quote(str(task))}",
            f"{_shell_quote(self.config['python'] or os.path.realpath(sys.executable))} fp_recovery_worker.py",
            "",
        ]
        return job_name, command, values, "\n".join(lines)

    # ---- attempts --------------------------------------------------------

    def _archive(self, name, label=None):
        """Move outputs of the previous attempt to attempts/NNN; never delete."""
        task = self.root / name
        entry = self.state["tasks"][name]
        present = [f for f in OUTPUT_FILES if (task / f).exists()]
        present += sorted(p.name for p in task.glob("slurm-*.out"))
        present += sorted(p.name for p in task.glob("fp-pilot-*.log"))
        if not present:
            return None
        if label is None:
            previous = entry["attempts"][-1]["n"] if entry["attempts"] else 0
            label = f"{previous:03d}"
        target = task / "attempts" / label
        if target.exists():
            target = task / "attempts" / f"{label}-{time.time_ns()}"
        if self.dry_run:
            return target
        target.mkdir(parents=True)
        for filename in present:
            os.replace(task / filename, target / filename)
        for filename in ("fp.sbatch", "fp-recovery.json"):
            if (task / filename).exists():
                shutil.copy2(task / filename, target / filename)
        return target

    def _health(self):
        if self.health_path.exists():
            return json.loads(self.health_path.read_text())
        return {"devices": {}}

    def submit(self, name, queue):
        entry = self.state["tasks"][name]
        task = self.root / name
        if self._reusable_outputs(name):
            self._mark_done(
                name,
                entry["attempts"][-1] if entry["attempts"] else None,
                "valid results found before resubmission",
            )
            return
        running_here = [
            job_id for job_id, job in queue.items() if _same_path(job["workdir"], task)
        ]
        if running_here:
            self._adopt_job(
                name,
                running_here[0],
                queue[running_here[0]]["state"],
                "found already queued in this task directory",
            )
            return
        number = (entry["attempts"][-1]["n"] if entry["attempts"] else 0) + 1
        archived = self._archive(name)
        job_name, command, values, script = self.render(name, number)
        if self.dry_run:
            self.scripts[name] = script
        else:
            shutil.copyfile(worker.__file__, task / "fp_recovery_worker.py")
            if self.monitor["enabled"]:
                shutil.copyfile(vasp_monitor.__file__, task / "fp_vasp_monitor.py")
            worker.atomic_json(
                task / "fp-recovery.json",
                {
                    "command": command,
                    "bad_gpu_uuids": sorted(self._health()["devices"]),
                    "fp_monitor": self.monitor,
                    "validated_input_digest": None,
                },
            )
            (task / "fp.sbatch").write_text(script)
        attempt = {
            "n": number,
            "job_id": None,
            "job_name": job_name,
            "vars": values,
            "command": command,
            "submitted_at": self.clock(),
            "archived_previous": str(archived.relative_to(task)) if archived else None,
        }
        entry["attempts"].append(attempt)
        entry["submitting"] = True
        self.save()
        try:
            job_id = self.slurm.submit("fp.sbatch", cwd=str(task))
        except SubmitRejected as exc:
            entry.pop("submitting", None)
            attempt.update(
                category="submit_rejected", reason=str(exc), ended=self.clock()
            )
            self._decide(name, attempt, "submit_rejected", f"sbatch rejected: {exc}")
            return
        except SubmitUncertain as exc:
            self.save()
            self.event(
                name,
                f"submission outcome unknown ({exc}); will look for the job before retrying",
                "warning",
            )
            return
        entry.pop("submitting", None)
        attempt["job_id"] = job_id
        entry.update(status="queued", reason="")
        self.event(name, f"submitted job {job_id} as {job_name} ({_fmt_vars(values)})")

    def _adopt_job(self, name, job_id, state, why):
        entry = self.state["tasks"][name]
        if entry["attempts"] and not entry["attempts"][-1].get("job_id"):
            attempt = entry["attempts"][-1]
        else:
            attempt = {
                "n": (entry["attempts"][-1]["n"] if entry["attempts"] else 0) + 1,
                "vars": {},
                "submitted_at": None,
                "adopted": True,
            }
            entry["attempts"].append(attempt)
        attempt["job_id"] = job_id
        entry.pop("submitting", None)
        entry["status"] = "queued" if state == "PENDING" else "running"
        self.event(name, f"tracking job {job_id} ({why})")

    def _resolve_uncertain(self, name, queue):
        entry = self.state["tasks"][name]
        attempt = entry["attempts"][-1]
        task = self.root / name
        found = [j for j, job in queue.items() if _same_path(job["workdir"], task)]
        if not found:
            try:
                found = self.slurm.find(task, (attempt.get("submitted_at") or 0) - 300)
            except SlurmError as exc:
                self.event(name, f"cannot check accounting yet: {exc}", "warning")
                return
        if found:
            state = queue.get(found[-1], {}).get("state", "COMPLETED")
            self._adopt_job(name, found[-1], state, "recovered after uncertain sbatch")
            if found[-1] not in queue:
                entry["status"] = "running"  # left the queue; finished next pass
            return
        waited = self.clock() - (attempt.get("submitted_at") or 0)
        if waited < self.config["accounting_grace"]:
            return
        entry.pop("submitting", None)
        attempt.update(category="not_submitted", reason="no job found after sbatch")
        entry.update(status="pending", retry_after=0)
        self.event(name, "no job was created by the uncertain sbatch; submitting again")

    def finish(self, name, accounting):
        entry = self.state["tasks"][name]
        attempt = entry["attempts"][-1]
        task = self.root / name
        attempt.update(
            ended=self.clock(),
            state=accounting.get("state"),
            exit_code=accounting.get("exit_code"),
            node=accounting.get("node"),
            partition=accounting.get("partition"),
            elapsed=accounting.get("elapsed"),
        )
        if attempt.get("legacy_dir"):
            self._harvest_legacy(name, attempt)
            task = Path(self.view.get(name, task))
        intent = attempt.get("cancel_intent")
        stop = self._monitor_stop(task, attempt.get("job_id"))
        if stop and intent != "discard":
            attempt.update(category="scientific_stop", reason=stop)
            entry.update(status="stopped", reason=stop)
            self.event(
                name, f"job {attempt['job_id']} stopped by FP monitor: {stop}; excluded"
            )
            return
        valid, reason = (False, "no OUTCAR")
        if (task / "OUTCAR").exists():
            valid, reason = self.validate(task)
        if valid:
            attempt.update(category="success", reason=reason)
            if entry.get("discard"):
                entry.update(status="discarded", reason=entry["discard"])
                self.event(
                    name, "completed before discard took effect; kept on disk, excluded"
                )
            else:
                self._mark_done(name, attempt, reason)
            return
        if intent:
            attempt.update(category="cancelled", reason=f"cancelled for {intent}")
            if intent == "discard":
                entry.update(
                    status="discarded", reason=entry.get("discard", "discarded")
                )
            elif intent == "hold":
                entry.update(status="held", reason="held by user")
            else:
                entry.update(
                    status="pending", retry_after=0, reason="resubmit requested"
                )
            self.event(
                name, f"job {attempt['job_id']} cancelled; now {entry['status']}"
            )
            return
        category, why = self._diagnose(task, attempt, accounting, reason)
        attempt.update(category=category, reason=why)
        self._decide(name, attempt, category, why)

    def _monitor_stop(self, task, job_id):
        try:
            stop = json.loads((task / "fp-monitor-stop.json").read_text())
            if str(stop.get("job_id")) == str(job_id):
                return stop.get("reason", "stopped by FP monitor")
        except (OSError, ValueError):
            pass
        report = self._attempt_report(task, job_id)
        if report.get("category") == "scientific_stop":
            return report.get("reason", "stopped by FP monitor")
        return None

    @staticmethod
    def _attempt_report(task, job_id):
        try:
            report = json.loads((task / "fp-attempt.json").read_text())
        except (OSError, ValueError):
            return {}
        if job_id and report.get("job_id") and str(report["job_id"]) != str(job_id):
            return {}
        return report

    def _diagnose(self, task, attempt, accounting, validation_reason):
        job_id = attempt.get("job_id")
        report = self._attempt_report(task, job_id)
        evidence = ""
        for filename in (
            "fp-runtime.log",
            f"slurm-{job_id}.out",
            f"fp-pilot-{job_id}.log",
            "fp.log",
        ):
            try:
                evidence += (task / filename).read_text(errors="replace")[
                    -16000:
                ] + "\n"
            except OSError:
                pass
        state = (accounting.get("state") or "").upper()
        if state.startswith("OUT_OF_ME"):
            return "resource", "Slurm reports OUT_OF_MEMORY"
        if state.startswith("CANCELLED"):
            return "cancelled", f"job cancelled outside the workflow ({state})"
        category, why = worker.classify(evidence, state)
        if report.get("category") in worker.UNLIMITED_RETRY_CATEGORIES:
            category, why = report["category"], report.get("reason", why)
        if category == "unknown" and (
            report.get("exit_code") == 0 or state.startswith("COMPLETED")
        ):
            category, why = "application", validation_reason
        return category, why

    def _decide(self, name, attempt, category, why):
        entry = self.state["tasks"][name]
        job = attempt.get("job_id") or "-"
        if category == "hardware":
            report = self._attempt_report(self.root / name, attempt.get("job_id"))
            faulty = report.get("faulty_gpu_uuids", [])
            if faulty:
                health = self._health()
                for uuid_ in faulty:
                    health["devices"][uuid_] = {
                        "node": report.get("node"),
                        "evidence": why,
                        "time": self.clock(),
                    }
                if not self.dry_run:
                    worker.atomic_json(self.health_path, health)
        if category == "cancelled":
            entry.update(status="held", reason=why)
            self.event(
                name,
                f"job {job}: {why}; held until you release or resubmit it",
                "warning",
            )
            return
        current = self.resolve_vars(name)
        used_vars = attempt.get("vars") or {}
        ladder = self.config["escalate"].get(category, {})
        raised, changes = {}, []
        for key, values in ladder.items():
            # Step up from what the failed job actually ran with; a rule or
            # override that is already higher is kept as it is.
            ran = used_vars.get(key, current.get(key))
            value = next_rung(ran, values)
            if value is None:
                continue
            target = _magnitude(value)
            already = _magnitude(current.get(key))
            if target is not None and already is not None and already >= target:
                changes.append(f"{key} {ran}→{current.get(key)}")
                continue
            raised[key] = value
            changes.append(f"{key} {ran}→{value}")
        if changes:
            entry["escalated"].update(raised)
            entry.update(status="pending", retry_after=0, reason=f"{category}: {why}")
            changes = ", ".join(changes)
            self.event(
                name, f"job {job} {category}: {why}; escalating {changes}", "warning"
            )
            return
        used = entry["retries"].get(category, 0)
        limit = self.config["retry"].get(category, self.config["retry"]["unknown"])
        if limit < 0 or used < limit:
            entry["retries"][category] = used + 1
            entry.update(
                status="pending",
                retry_after=self.clock() + self.config["retry_delay"],
                reason=f"{category}: {why}",
            )
            self.event(
                name,
                f"job {job} {category}: {why}; retry {used + 1}"
                + ("" if limit < 0 else f"/{limit}"),
                "warning",
            )
            return
        status = "held" if self.config["on_exhausted"] == "hold" else "failed"
        entry.update(status=status, reason=f"{category}: {why}")
        hint = (
            f" — e.g. `dpgen fp set {self.root} --tasks {name} --var KEY=VALUE --resubmit`"
            f" or `dpgen fp discard {self.root} --tasks {name} --reason ...`"
            if status == "held"
            else ""
        )
        self.event(name, f"job {job} {category}: {why}; {status}{hint}", "error")

    # ---- main loop -------------------------------------------------------

    def step(self):
        self.apply_requests()
        queue = self.slurm.queue()
        tasks = self.state["tasks"]
        tracked = {
            str(e["attempts"][-1].get("job_id")): name
            for name, e in tasks.items()
            if e["attempts"] and e["status"] in ACTIVE
        }
        for job_id, job in queue.items():
            owner = next(
                (n for n in tasks if _same_path(job["workdir"], self.root / n)), None
            )
            if (
                owner
                and tracked.get(job_id) != owner
                and not tasks[owner].get("submitting")
            ):
                if tasks[owner].get("foreign_warned") != job_id:
                    tasks[owner]["foreign_warned"] = job_id
                    self.event(
                        owner,
                        f"untracked job {job_id} is active in this directory; it is left "
                        "alone and no other job is submitted here while it runs",
                        "warning",
                    )
        finished = []
        for name, entry in tasks.items():
            if entry.get("submitting"):
                self._resolve_uncertain(name, queue)
                continue
            if entry["status"] in ACTIVE:
                if entry["attempts"][-1].get("allocation"):
                    continue  # tracked through its allocation
                job_id = str(entry["attempts"][-1].get("job_id"))
                if job_id in queue:
                    pending = queue[job_id]["state"] == "PENDING"
                    entry["status"] = "queued" if pending else "running"
                    entry.pop("left_queue_at", None)
                    attempt = entry["attempts"][-1]
                    if not pending and attempt.get("cancel_intent") == "update":
                        # It started before the pending-only cancel reached it:
                        # let it run and judge its outcome normally.
                        attempt.pop("cancel_intent")
                        self.event(
                            name,
                            f"job {job_id} started before it could be updated; "
                            "it keeps its original resources",
                        )
                    elif pending:
                        self._refresh_pending(name, attempt)
                else:
                    finished.append(name)
        if finished:
            accounting = self.slurm.accounting(
                [str(tasks[n]["attempts"][-1].get("job_id")) for n in finished]
            )
            for name in finished:
                entry = tasks[name]
                job_id = str(entry["attempts"][-1].get("job_id"))
                row = accounting.get(job_id)
                if row is None:
                    entry.setdefault("left_queue_at", self.clock())
                    if (
                        self.clock() - entry["left_queue_at"]
                        < self.config["accounting_grace"]
                    ):
                        continue
                    row = {"state": "UNKNOWN"}
                elif (
                    row["state"]
                    .upper()
                    .startswith(("RUNNING", "PENDING", "REQUEUED", "COMPLETING"))
                ):
                    continue
                entry.pop("left_queue_at", None)
                self.finish(name, row)
                self.save()
        if self.config["pack"]["enabled"]:
            self._pack().step(queue)
        else:
            if any(
                a["status"] in ("submitting", "queued", "running")
                for _, pool in fp_pack.stage_pools(self.root)
                for a in pool["allocations"].values()
            ):
                # Pack mode was switched off: finish tracking what runs there.
                self._pack().update(queue)
                self._pack().pool.save()
            for name, entry in tasks.items():
                if entry["status"] == "pending" and not entry.get("submitting"):
                    if entry.get("hold"):
                        entry.update(status="held", reason="held by user")
                        continue
                    if self.clock() >= entry.get("retry_after", 0):
                        self.submit(name, queue)
        self.save()

    # ---- dpgen-live.log / dpgen-summary.log -------------------------------

    def report(self):
        """One live line per check; a summary line at 25/50/75 % and at the end."""
        if self.dry_run:
            return
        from dpgen import progress

        workdir = self.root.parent.parent
        tasks = self.state["tasks"]
        counts = self.summary()
        total = len(tasks)
        finished = sum(counts.get(s, 0) for s in TERMINAL)
        order = (
            "done",
            "running",
            "queued",
            "pending",
            "held",
            "failed",
            "stopped",
            "discarded",
        )
        text = " ".join(f"{k} {counts[k]}" for k in order if counts.get(k))
        nodes = ""
        if self.config["pack"]["enabled"] or self.pack is not None:
            nodes = " | " + self._pack().pool.live_text()
        progress.live(
            workdir,
            f"FP {self.iter_label} | {finished}/{total} finished: {text}{nodes}",
        )
        marks = self.state.setdefault("milestones", [])
        for mark in (25, 50, 75):
            if (
                total
                and finished * 100 >= mark * total
                and mark not in marks
                and finished < total
            ):
                marks.append(mark)
                progress.summary(
                    workdir,
                    "fp_progress",
                    f"{progress.iter_label(int(self.iter_label[1:]))}  FP {mark}%: {finished}/{total} finished ({text})",
                    now=self.clock(),
                    iteration=self.iter_label,
                    percent=mark,
                    **counts,
                )
        if self.complete() and "complete" not in marks:
            marks.append("complete")
            reasons = {}
            for name, entry in tasks.items():
                if entry["status"] in ("failed", "stopped", "discarded"):
                    attempt = entry["attempts"][-1] if entry["attempts"] else {}
                    why = attempt.get("category") or entry["status"]
                    reasons[f"{entry['status']}: {why}"] = (
                        reasons.get(f"{entry['status']}: {why}", 0) + 1
                    )
            elapsed = self.clock() - self.state.get("created", self.clock())
            lines = [
                f"{progress.iter_label(int(self.iter_label[1:]))}  FP complete: "
                f"{counts.get('done', 0)} of {total} done in {progress.duration(elapsed)}"
            ]
            lines += [
                f"  {count} {reason}" for reason, count in sorted(reasons.items())
            ]
            progress.summary(
                workdir,
                "fp_complete",
                lines,
                now=self.clock(),
                iteration=self.iter_label,
                seconds=round(elapsed),
                reasons=reasons,
                **counts,
            )
        self.save()

    def _pack(self):
        if self.pack is None:
            self.pack = fp_pack.PackManager(self, self.config["pack"])
        return self.pack

    def complete(self):
        return all(e["status"] in TERMINAL for e in self.state["tasks"].values())

    def summary(self):
        counts = {}
        for entry in self.state["tasks"].values():
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        return counts

    def wait(self):
        path = self.root / REQUESTS
        try:
            seen = path.stat().st_mtime_ns
        except OSError:
            seen = None
        deadline = self.clock() + self.config["check_interval"]
        while self.clock() < deadline:
            self.sleep(min(2.0, max(0.0, deadline - self.clock())))
            try:
                if (
                    path.stat().st_mtime_ns != seen
                    and read_requests(self.root)["requests"]
                ):
                    return
            except (OSError, ValueError):
                pass

    def run(self):
        with ExitStack() as stack:
            stack.enter_context(_flock(self.root / ".fp-campaign.lock", blocking=False))
            # Keep any older recovery controller out of this stage.
            stack.enter_context(_flock(self.root / ".fp-recovery.lock", blocking=False))
            self.initialize(self.slurm.queue())
            last = None
            while True:
                self.step()
                self.report()
                if self.complete():
                    break
                counts = self.summary()
                if counts != last:
                    held = counts.get("held", 0)
                    dlog.info(
                        "FP %s: %s%s; status table %s",
                        self.iter_label,
                        " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                        f" — {held} held task(s) need `dpgen fp` action"
                        if held
                        else "",
                        self.root / STATUS,
                    )
                    last = counts
                self.wait()
            counts = self.summary()
            self.event(
                None,
                "stage complete: "
                + " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
            )
            return counts


def _stop_reason(directory):
    """Reason recorded by the VASP monitor in ``directory``, if it stopped the run."""
    directory = Path(directory)
    try:
        stop = json.loads((directory / "fp-monitor-stop.json").read_text())
        return stop.get("reason") or "stopped by the FP monitor"
    except (OSError, ValueError):
        pass
    try:
        report = json.loads((directory / "fp-attempt.json").read_text())
    except (OSError, ValueError):
        return None
    if report.get("category") == "scientific_stop":
        return report.get("reason") or "stopped by the FP monitor"
    return None


def _previous_failure(directory, old):
    """Short description of why the previous controller's attempt failed."""
    try:
        report = json.loads((Path(directory) / "fp-attempt.json").read_text())
    except (OSError, ValueError):
        report = {}
    diagnosis = old.get("diagnosis") or {}
    category = report.get("category") or diagnosis.get("category")
    reason = report.get("reason") or old.get("reason") or diagnosis.get("reason")
    if category in (None, "success") and not reason:
        return None
    return ": ".join(str(x) for x in (category, reason) if x)[:200]


def _rule_matches(rule, name, entry):
    if "min_natoms" in rule and entry["natoms"] < rule["min_natoms"]:
        return False
    if "max_natoms" in rule and entry["natoms"] > rule["max_natoms"]:
        return False
    if "systems" in rule and int(entry.get("sys") or -1) not in {
        int(s) for s in rule["systems"]
    }:
        return False
    if "tasks" in rule and not any(fnmatch(name, pattern) for pattern in rule["tasks"]):
        return False
    return True


def _fmt_vars(values):
    return " ".join(f"{k}={v}" for k, v in sorted(values.items())) or "-"


def _shell_quote(text):
    import shlex

    return shlex.quote(text)


def status_rows(state, config=None):
    rows = []
    for name, entry in sorted(state["tasks"].items()):
        attempt = entry["attempts"][-1] if entry.get("attempts") else {}
        rows.append(
            {
                "task": name,
                "natoms": entry.get("natoms", ""),
                "formula": entry.get("formula", ""),
                "status": entry["status"]
                + (
                    " (hold)" if entry.get("hold") and entry["status"] != "held" else ""
                ),
                "attempt": attempt.get("n", 0),
                "job_id": (attempt.get("job_id") or "-")
                + (f" ({attempt['allocation']})" if attempt.get("allocation") else ""),
                "resources": _fmt_vars(attempt.get("vars") or {}),
                "reason": (entry.get("reason") or "")
                .replace("\t", " ")
                .replace("\n", " ")[:160],
            }
        )
    return rows


def attach_pool(config, stage_root, mdata, write):
    """With pack.shared, use machine.json ``pool`` (the run-wide node pool)."""
    if not config["pack"].get("shared"):
        return config
    if mdata.get("pool") is None:
        raise ValueError(
            'fp.slurm.pack.shared needs a top-level "pool" section in machine.json'
        )
    workdir = Path(stage_root).resolve().parent.parent
    config["pool"] = (
        fp_pack.node_pool.configure(workdir, mdata)
        if write
        else fp_pack.node_pool.pool_options(mdata["pool"])
    )
    return config


def run_campaign(work_path, jdata, mdata, slurm=None, moved=False):
    if jdata.get("fp_style") != "vasp":
        raise ValueError("fp.slurm currently supports fp_style='vasp' only")
    config = attach_pool(options(mdata.get("fp_slurm")), work_path, mdata, write=True)
    if not config["enabled"]:
        raise RuntimeError("fp.slurm must be enabled to resume this FP campaign")
    counts = Campaign(work_path, config, jdata, slurm=slurm, moved=moved).run()
    if counts.get("failed"):
        dlog.warning(
            "FP: %d task(s) failed and are excluded from labels; ratio_failed applies",
            counts["failed"],
        )
    return counts


STEP_NAMES = (
    "make_train",
    "run_train",
    "post_train",
    "make_model_devi",
    "run_model_devi",
    "post_model_devi",
    "make_fp",
    "run_fp",
    "post_fp",
)


def plan_stage(
    root, jdata, mdata, slurm=None, out=sys.stdout, validator=None, moved=False
):
    """Print what the controller would do for one FP stage; change nothing.

    Returns 0 when the stage would proceed and 1 when it would stop.
    """
    try:
        config = attach_pool(options(mdata.get("fp_slurm")), root, mdata, write=False)
    except ValueError as exc:
        print(f"ERROR in machine.json: {exc}", file=out)
        return 1
    campaign = Campaign(
        root,
        config,
        jdata,
        slurm=slurm or Slurm(),
        validator=validator,
        dry_run=True,
        moved=moved,
    )
    print(f"\n== FP plan: {campaign.root} ==", file=out)
    if not campaign._task_dirs():
        print("No task.* directories: run_fp would have nothing to do.", file=out)
        return 0
    mode = (
        "take over a stage copied from another cluster"
        if moved and (campaign.root / STATE).exists()
        else "resume existing campaign"
        if (campaign.root / STATE).exists()
        else "adopt previous recovery controller's jobs"
        if (campaign.root / LEGACY_LEDGER).exists()
        else "new campaign"
    )
    print(f"Mode: {mode}", file=out)
    try:
        campaign.initialize(campaign.slurm.queue())
        campaign.step()
    except (RuntimeError, ValueError, SlurmError) as exc:
        print(f"\nWOULD STOP before submitting anything:\n  {exc}", file=out)
        return 1

    if campaign.plan:
        print("\nWhat would happen, in order:", file=out)
        for label, message, level in campaign.plan:
            mark = "!" if level in ("warning", "error") else "-"
            print(f"  {mark} {label}: {message}", file=out)
    else:
        print("\nNo changes: nothing to adopt, update, submit or cancel.", file=out)

    cancelled = campaign.slurm.cancelled
    if cancelled:
        print(f"\nWould cancel {len(cancelled)} job(s):", file=out)
        for job_id, pending_only in cancelled:
            note = (
                " (only if still pending; a started job is left running)"
                if pending_only
                else ""
            )
            print(f"  scancel {job_id}{note}", file=out)

    submitted = [name for name in campaign.scripts]
    if submitted:
        print(f"\nWould submit {len(submitted)} job(s):", file=out)
        rows = []
        for name in submitted:
            entry = campaign.state["tasks"][name]
            rows.append(
                {
                    "task": name,
                    "natoms": entry["natoms"],
                    "formula": entry["formula"],
                    "resources": _fmt_vars(entry["attempts"][-1]["vars"]),
                }
            )
        widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in rows[0]}
        print("  " + "  ".join(k.upper().ljust(widths[k]) for k in rows[0]), file=out)
        for row in rows:
            print("  " + "  ".join(str(row[k]).ljust(widths[k]) for k in row), file=out)
        samples = {}
        for name in submitted:
            key = _fmt_vars(campaign.state["tasks"][name]["attempts"][-1]["vars"])
            samples.setdefault(key, name)
        for key, name in samples.items():
            print(
                f"\n--- fp.sbatch that would be written for {name} ({key}) ---",
                file=out,
            )
            print(campaign.scripts[name].rstrip(), file=out)

    pack = campaign.config["pack"]
    if pack["enabled"]:
        limits = campaign._pack().limits
        print(
            f"\nAllocations{' (run-wide pool, shared with training/exploration)' if pack['shared'] else ''}: "
            f"up to {campaign._pack().pool.limit('fp')} × {limits['gpus']} GPU for "
            f"{limits['time']}; tasks are assigned only to allocations that have started.",
            file=out,
        )
        allocs = campaign._pack().allocs
        rows = fp_pack.summary_rows(allocs)
        if rows:
            for row in rows:
                print(
                    f"  {row['allocation']}  job {row['job_id']:<10} {row['status']:<10} "
                    f"{row['gpus']:<10} left {row['time_left']:<7} {row['tasks']}",
                    file=out,
                )
        waiting = [
            n for n, e in campaign.state["tasks"].items() if e["status"] == "pending"
        ]
        if waiting:
            print(
                f"\n{len(waiting)} task(s) wait for a running allocation. Estimates:",
                file=out,
            )
            for name in sorted(waiting)[:20]:
                entry = campaign.state["tasks"][name]
                value, basis, _ = campaign._pack().estimate(name)
                print(
                    f"  {name}  {entry['natoms']:>5} at  {campaign._pack().ngpus(name)} GPU  "
                    f"{fp_pack.hours(value):>7}  ({basis})",
                    file=out,
                )
            if len(waiting) > 20:
                print(f"  … and {len(waiting) - 20} more", file=out)
        command = campaign.config["command"]
        if "mpirun" in command and "bind-to" not in command:
            print(
                "\n! fp.slurm.command uses mpirun without --bind-to. With several tasks "
                "in one allocation, Open MPI binds every task's ranks to the same cores. "
                "Use `mpirun --bind-to none -np {ngpus} ...`.",
                file=out,
            )
        for alloc_id, script in campaign.alloc_scripts.items():
            print(
                f"\n--- alloc.sbatch that would be written for allocation {alloc_id} ---",
                file=out,
            )
            print(script.rstrip(), file=out)
        del allocs

    counts = campaign.summary()
    print(
        "\nAfter this first pass: "
        + " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        file=out,
    )
    print("Dry run: nothing was submitted, cancelled, moved or written.", file=out)
    return 0


def _record_next_step(record="record.dpgen"):
    if not os.path.isfile(record):
        return None
    last = None
    with open(record) as stream:
        for line in stream:
            if line.strip():
                last = [int(x) for x in line.split()]
    if last is None:
        return None
    iteration, step = last
    return (iteration, step + 1) if step < 8 else (iteration + 1, 0)


def workflow_dry_run(
    param_file, machine_file, out=sys.stdout, slurm=None, validator=None
):
    """``dpgen run|finetune PARAM MACHINE --dry-run``: check settings, plan FP."""
    from dpgen.generator.lib.utils import make_iter_name
    from dpgen.remote.decide_machine import convert_mdata
    from dpgen.util import load_file

    print("dpgen dry run: nothing is submitted, cancelled or written.", file=out)
    jdata = load_file(param_file)
    mdata = convert_mdata(load_file(machine_file))
    resources = mdata.get("fp_resources") or {}
    if isinstance(resources, dict) and "slurm" in resources:
        print(
            'ERROR: machine.json has "slurm" inside fp.resources. Move it up one '
            'level so it sits next to "resources" inside "fp".',
            file=out,
        )
        return 1
    if not enabled(mdata):
        print(
            "fp.slurm is not enabled: FP jobs would be submitted by the previous "
            "dispatcher path. Nothing to plan.",
            file=out,
        )
        return 0
    try:
        config = options(mdata["fp_slurm"])
    except ValueError as exc:
        print(f"ERROR in machine.json fp.slurm: {exc}", file=out)
        return 1
    print("fp.slurm settings: valid", file=out)
    print(f"  base resources: {_fmt_vars(config['vars'])}", file=out)
    for rule in config["rules"]:
        when = ", ".join(f"{k}={v}" for k, v in rule.items() if k != "vars")
        print(f"  rule: {when or 'all tasks'} -> {_fmt_vars(rule['vars'])}", file=out)
    for category, ladder in config["escalate"].items():
        steps = "; ".join(f"{k}: {' -> '.join(map(str, v))}" for k, v in ladder.items())
        print(f"  on {category}: {steps}", file=out)
    print(
        f"  pending jobs follow changes: {'yes' if config['update_pending'] else 'no'}; "
        f"when retries run out: {config['on_exhausted']}",
        file=out,
    )
    if jdata.get("fp_style") != "vasp":
        print(
            f"ERROR: fp.slurm supports fp_style 'vasp', not {jdata.get('fp_style')!r}",
            file=out,
        )
        return 1

    nxt = _record_next_step()
    if nxt is None:
        print(
            "\nrecord.dpgen not found: the workflow has not started. FP is planned once its tasks exist.",
            file=out,
        )
        return 0
    iteration, step = nxt
    print(
        f"\nNext workflow step: {make_iter_name(iteration)} step {step:02d} ({STEP_NAMES[step]})",
        file=out,
    )
    stage = Path(make_iter_name(iteration)) / "02.fp"
    if step < 7:
        print(
            f"FP tasks for {make_iter_name(iteration)} are not prepared yet (make_fp is step 06); "
            "training/exploration would run first. Run the dry run again once FP is next.",
            file=out,
        )
        return 0
    if step == 8:
        print("FP for this iteration has finished; post_fp would run next.", file=out)
    if not stage.is_dir():
        print(f"{stage} does not exist.", file=out)
        return 0
    return plan_stage(stage, jdata, mdata, slurm=slurm, out=out, validator=validator)


def excluded(task):
    """True when an in-place campaign exists and this task is not a validated result."""
    task = Path(task)
    try:
        state = json.loads((task.parent / STATE).read_text())
    except (OSError, ValueError):
        return False
    entry = state.get("tasks", {}).get(task.name)
    return entry is not None and entry.get("status") != "done"


# --------------------------------------------------------------------------
# command line: dpgen fp ...


def find_stage(value=None):
    if value:
        path = Path(value).resolve()
        if path.name != "02.fp" and (path / "02.fp").is_dir():
            path = path / "02.fp"
        if not (path / STATE).exists():
            raise SystemExit(
                f"No {STATE} in {path}; is fp.slurm enabled for this stage?"
            )
        return path
    stages = sorted(Path.cwd().glob(f"iter.*/02.fp/{STATE}"))
    if not stages:
        raise SystemExit(f"No FP campaign ({STATE}) found under {Path.cwd()}")
    return stages[-1].parent.resolve()


def select_tasks(state, args):
    tasks = state["tasks"]
    names = set()
    explicit = False
    if args.tasks:
        explicit = True
        for pattern in args.tasks:
            matched = [
                n for n in tasks if fnmatch(n, pattern) or fnmatch(n, f"task.{pattern}")
            ]
            if not matched:
                raise SystemExit(f"No task matches {pattern!r}")
            names.update(matched)
    if args.jobs:
        explicit = True
        for job_id in args.jobs:
            matched = [
                n
                for n, e in tasks.items()
                if any(
                    str(a.get("job_id")) == str(job_id) for a in e.get("attempts", [])
                )
            ]
            if not matched:
                raise SystemExit(f"Job {job_id} does not belong to this FP stage")
            names.update(matched)
    if not explicit:
        if not (
            args.status or args.min_natoms is not None or args.max_natoms is not None
        ):
            raise SystemExit(
                "Select tasks with --tasks, --jobs, --status, --min-natoms or --max-natoms"
            )
        names = set(tasks)
    if args.status:
        wanted = {s for value in args.status for s in value.split(",")}
        names = {n for n in names if tasks[n]["status"] in wanted}
    if args.min_natoms is not None:
        names = {n for n in names if tasks[n]["natoms"] >= args.min_natoms}
    if args.max_natoms is not None:
        names = {n for n in names if tasks[n]["natoms"] <= args.max_natoms}
    done = sorted(n for n in names if tasks[n]["status"] == "done")
    if done and explicit and args.tasks:
        raise SystemExit(
            f"Refusing to change task(s) with validated results: {', '.join(done)}"
        )
    return sorted(n for n in names if tasks[n]["status"] != "done"), done


def _set_effect(entry, resubmit):
    status = entry["status"]
    if status == "queued":
        return (
            "cancel + resubmit now"
            if resubmit
            else "pending job replaced before it starts (update_pending)"
        )
    if status == "running":
        return (
            "cancel running job + resubmit"
            if resubmit
            else "running job untouched; applies to the next attempt"
        )
    if status in ("held", "failed", "stopped"):
        return "resubmit now" if resubmit else "applies when released or resubmitted"
    return "applies to the next submission"


def queue_request(root, action, tasks, **fields):
    request = {
        "id": uuid.uuid4().hex,
        "action": action,
        "tasks": tasks,
        "created": time.time(),
        "user": getpass.getuser(),
        **fields,
    }
    with _flock(Path(root) / ".fp-requests.lock"):
        queue = read_requests(root)
        queue.setdefault("requests", []).append(request)
        worker.atomic_json(Path(root) / REQUESTS, queue)
    return request


def _parse_vars(items):
    values = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--var expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in META_KEYS:
            raise SystemExit(f"Invalid variable name {key!r}")
        values[key] = int(value) if re.fullmatch(r"-?\d+", value) else value
    return values


def _print_table(rows, out=sys.stdout):
    columns = (
        "task",
        "natoms",
        "formula",
        "status",
        "attempt",
        "job_id",
        "resources",
        "reason",
    )
    widths = {c: max([len(c)] + [len(str(r[c])) for r in rows]) for c in columns[:-1]}
    print("  ".join(c.upper().ljust(widths.get(c, 0)) for c in columns), file=out)
    for row in rows:
        print("  ".join(str(row[c]).ljust(widths.get(c, 0)) for c in columns), file=out)


def _stage_arg(value):
    here = Path.cwd()
    if not value and here.name != "02.fp" and not (here / "02.fp").is_dir():
        return find_stage(None)  # latest iter.*/02.fp with a campaign
    stage = Path(value or here).resolve()
    if stage.name != "02.fp" and (stage / "02.fp").is_dir():
        stage = stage / "02.fp"
    if not stage.is_dir():
        raise SystemExit(f"{stage} is not a directory")
    return stage


def command(args):
    if args.fp_action in ("plan", "run"):
        from dpgen.remote.decide_machine import convert_mdata
        from dpgen.util import load_file

        stage = _stage_arg(args.STAGE)
        jdata = load_file(args.param)
        mdata = convert_mdata(load_file(args.machine))
        moved = getattr(args, "moved", False)
        if args.fp_action == "plan" or args.dry_run:
            raise SystemExit(plan_stage(stage, jdata, mdata, moved=moved))
        counts = run_campaign(stage, jdata, mdata, moved=moved)
        print(
            f"FP stage {stage} complete: "
            + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        )
        return
    root = find_stage(args.STAGE)
    state = json.loads((root / STATE).read_text())
    if args.fp_action == "status":
        rows = status_rows(state)
        if args.status:
            wanted = {s for value in args.status for s in value.split(",")}
            rows = [r for r in rows if r["status"].split(" ")[0] in wanted]
        if args.min_natoms is not None:
            rows = [r for r in rows if r["natoms"] >= args.min_natoms]
        _print_table(rows)
        counts = {}
        for entry in state["tasks"].values():
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        print(
            "\n"
            + root.as_posix()
            + ": "
            + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        )
        allocations = [
            row
            for _, pool in fp_pack.stage_pools(root)
            for row in fp_pack.summary_rows(pool["allocations"])
        ]
        if allocations:
            print(
                "\nALLOCATION  JOB_ID      STATUS     NODE        GPUS        LEFT     TASKS"
            )
            for row in allocations:
                print(
                    f"{row['allocation']:<10}  {row['job_id']:<10}  {row['status']:<9}  "
                    f"{row['node']:<10}  {row['gpus']:<10}  {row['time_left']:<7}  {row['tasks']}"
                )
        pending = read_requests(root)["requests"]
        if pending:
            print(
                f"{len(pending)} request(s) waiting for the controller to apply them."
            )
        return
    tasks, skipped = select_tasks(state, args)
    if not tasks:
        raise SystemExit("No unfinished task matches the selection")
    fields = {}
    if args.fp_action == "set":
        fields["vars"] = _parse_vars(args.var)
        if not fields["vars"]:
            raise SystemExit("set needs at least one --var KEY=VALUE")
        fields["resubmit"] = args.resubmit
    if args.fp_action == "hold":
        fields["cancel"] = args.cancel
    if args.fp_action == "discard":
        if not (args.reason or "").strip():
            raise SystemExit("discard needs --reason")
        fields["reason"] = args.reason.strip()
    for name in tasks:
        entry = state["tasks"][name]
        line = f"{name}  {entry['natoms']:>5} at  {entry['formula']:<16} {entry['status']:<9}"
        if args.fp_action == "set":
            last = (
                entry["attempts"][-1].get("vars") if entry["attempts"] else None
            ) or {}
            changes = ", ".join(
                f"{k} {last.get(k, '?')}->{v}"
                for k, v in fields["vars"].items()
                if last.get(k) != v
            )
            line += f"  {changes or 'no change'}  " + _set_effect(
                entry, fields["resubmit"]
            )
        print(line)
    if skipped:
        print(f"(skipped {len(skipped)} task(s) with validated results)")
    if args.dry_run:
        print("Dry run: nothing queued.")
        return
    queue_request(root, args.fp_action, tasks, **fields)
    print(
        f"Queued {args.fp_action} for {len(tasks)} task(s). A running controller applies it "
        "within a few seconds; otherwise it is applied when dpgen resumes."
    )


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "fp",
        help="Inspect and steer in-place Slurm FP jobs (status, set, resubmit, ...).",
    )
    actions = parser.add_subparsers(dest="fp_action", required=True)

    def selectors(sub, required=True):
        sub.add_argument(
            "STAGE",
            nargs="?",
            help="iter.XXXXXX or iter.XXXXXX/02.fp (default: latest)",
        )
        sub.add_argument(
            "--tasks",
            nargs="+",
            default=[],
            help="task names or globs, e.g. task.000.0001*",
        )
        sub.add_argument("--jobs", nargs="+", default=[], help="Slurm job IDs")
        sub.add_argument(
            "--status", nargs="+", default=[], help="e.g. held failed running"
        )
        sub.add_argument("--min-natoms", type=int)
        sub.add_argument("--max-natoms", type=int)
        if required:
            sub.add_argument(
                "--dry-run", action="store_true", help="show the selection only"
            )

    plan = actions.add_parser(
        "plan", help="Show what the controller would do for a stage (changes nothing)"
    )
    plan.add_argument("STAGE", nargs="?")
    plan.add_argument("--machine", required=True, help="machine.json")
    plan.add_argument("--param", required=True, help="param.json")
    plan.add_argument(
        "--moved",
        action="store_true",
        help="preview taking over a stage copied from another cluster",
    )
    run = actions.add_parser(
        "run",
        help="Run the controller for one FP stage until every task is finished "
        "(no other workflow step), e.g. on another cluster",
    )
    run.add_argument("STAGE", nargs="?", help="iter.XXXXXX or iter.XXXXXX/02.fp")
    run.add_argument("--machine", required=True, help="machine.json for THIS cluster")
    run.add_argument("--param", required=True, help="param.json")
    run.add_argument(
        "--moved",
        action="store_true",
        help="the stage was copied here from another cluster: forget the job IDs "
        "there, keep results that validate, run the unfinished tasks here",
    )
    run.add_argument(
        "--dry-run", action="store_true", help="show what would happen; change nothing"
    )
    status = actions.add_parser("status", help="Show a table of FP tasks")
    status.add_argument("STAGE", nargs="?")
    status.add_argument("--status", nargs="+", default=[])
    status.add_argument("--min-natoms", type=int)
    sub = actions.add_parser(
        "set", help="Override resource variables, e.g. --var ngpus=2"
    )
    selectors(sub)
    sub.add_argument("--var", action="append", default=[], metavar="KEY=VALUE")
    sub.add_argument(
        "--resubmit", action="store_true", help="cancel active jobs and resubmit now"
    )
    sub = actions.add_parser("resubmit", help="Cancel (if active) and resubmit now")
    selectors(sub)
    sub = actions.add_parser("hold", help="Do not submit these tasks again")
    selectors(sub)
    sub.add_argument("--cancel", action="store_true", help="also cancel active jobs")
    sub = actions.add_parser(
        "release", help="Allow held/failed/stopped tasks to run again"
    )
    selectors(sub)
    sub = actions.add_parser("discard", help="Cancel and exclude tasks from labels")
    selectors(sub)
    sub.add_argument("--reason", required=True)
    parser.set_defaults(func=command)


def _main(argv=None):
    parser = argparse.ArgumentParser(prog="dpgen")
    add_parser(parser.add_subparsers())
    args = parser.parse_args(argv)
    args.func(args)
