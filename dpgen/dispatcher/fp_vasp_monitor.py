"""Standalone, opt-in watchdog for VASP single-point electronic minimization."""

import json
import math
import os
import re
import signal
import threading
import time
from pathlib import Path

DEFAULTS = {
    "enabled": False,
    "check_interval": 10,
    "max_scf_steps": 200,
    "startup_timeout": 1800,
    "stall_timeout": 1800,
    "max_runtime": 0,
    "stop_grace_period": 120,
    "kill_grace_period": 10,
    "min_scf_steps": 60,
    "stagnation_window": 0,
    "min_relative_improvement": 0.05,
    "unconverged_factor": 100,
}


def options(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or value.keys() - DEFAULTS.keys():
        raise ValueError("Invalid fp_monitor options")
    result = dict(DEFAULTS, **value)
    if not isinstance(result["enabled"], bool):
        raise ValueError("fp_monitor.enabled must be boolean")
    for key in DEFAULTS.keys() - {"enabled"}:
        number = result[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number < 0
        ):
            raise ValueError(f"Invalid fp_monitor.{key}")
    for key in ("max_scf_steps", "min_scf_steps", "stagnation_window"):
        if int(result[key]) != result[key]:
            raise ValueError(f"fp_monitor.{key} must be an integer")
    if result["check_interval"] <= 0 or result["unconverged_factor"] < 1:
        raise ValueError(
            "fp_monitor requires a positive check_interval and unconverged_factor >= 1"
        )
    if result["min_relative_improvement"] >= 1:
        raise ValueError("fp_monitor.min_relative_improvement must be below 1")
    return result


def stopped_task(task):
    """Never label a monitor-stopped task, even after partial output transfer."""
    task = Path(task)
    if (task / "fp-monitor-stop.json").exists() or (task / "fp-discard.json").exists():
        return True
    try:
        if (
            json.loads((task / "fp-attempt.json").read_text()).get("category")
            == "scientific_stop"
        ):
            return True
    except (OSError, ValueError):
        pass
    try:
        ledger = json.loads((task.parent / ".fp-recovery.json").read_text())
        entry = ledger.get("tasks", {}).get(task.name, {})
        if entry.get("status") in ("discarded", "discard_pending"):
            return True
        queue = json.loads((task.parent / ".fp-discard-requests.json").read_text())
        request = queue.get("requests", {}).get(task.name, {})
        return (
            bool(request)
            and request.get("status") != "error"
            and request.get("submission_hash") == ledger.get("submission_hash")
            and request.get("input_digest") == entry.get("input_digest")
        )
    except (OSError, ValueError):
        return False


def incar_settings(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        line = re.split(r"[!#]", line)[0]
        for match in re.finditer(
            r"\b(NELM|NELMIN|EDIFF|NSW)\s*=\s*([^;\s]+)", line, re.I
        ):
            values[match[1].upper()] = float(
                match[2].replace("d", "e").replace("D", "E")
            )
    if values.get("NSW", 0) > 1:
        raise ValueError(
            "fp_monitor supports VASP single-point calculations only (NSW <= 1)"
        )
    if not math.isfinite(values.get("EDIFF", 1e-4)) or values.get("EDIFF", 1e-4) <= 0:
        raise ValueError("fp_monitor requires finite EDIFF > 0")
    if values.get("NELM", 60) < 1 or not math.isfinite(values.get("NELM", 60)):
        raise ValueError("fp_monitor requires positive NELM")
    return values


def scf_record(line):
    match = re.match(r"\s*(DAV|RMM(?:-DIIS)?|CG|DMP)\s*:\s*(\d+)\s+(.*)", line, re.I)
    if not match:
        return None
    fields = match[3].split()
    if len(fields) < 3:
        return None

    def number(text):
        if "*" in text:
            return float("nan")
        # Fortran sometimes omits E in large exponents.
        text = (
            re.sub(r"(?<=\d)([+-]\d{3,})$", r"e\1", text)
            if "e" not in text.lower() and "d" not in text.lower()
            else text
        )
        return float(text.replace("D", "E").replace("d", "e"))

    try:
        result = dict(
            step=int(match[2]),
            energy=number(fields[0]),
            de=number(fields[1]),
            deps=number(fields[2]),
        )
        if len(fields) > 4:
            result["rms"] = number(fields[4])
        if len(fields) > 5:
            result["charge_rms"] = number(fields[5])
        return result
    except ValueError:
        return None


class Monitor:
    """Watch live stdout and OSZICAR; STOPCAR first, own process group last."""

    def __init__(self, policy, root=".", clock=time.monotonic):
        self.policy = options(policy)
        self.root = Path(root)
        self.incar = incar_settings(self.root / "INCAR")
        self.clock = clock
        self.started = self.activity = clock()
        self.records = []
        self.last_step = 0
        self.reason = None
        self.stopped_at = None
        self.termination = None
        self.process = None
        self.lock = threading.RLock()
        self.done = threading.Event()
        self.thread = None
        self.sizes = {}
        for name in ("OUTCAR", "OSZICAR"):
            try:
                stat = (self.root / name).stat()
                self.sizes[name] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            except OSError:
                self.sizes[name] = None
        self.oszicar_position = self.sizes["OSZICAR"][1] if self.sizes["OSZICAR"] else 0
        self.partial = b""

    def start(self, process):
        self.process = process
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def feed(self, line):
        with self.lock:
            self.activity = self.clock()
            record = scf_record(line)
            if record is None or record["step"] <= self.last_step:
                return
            self.last_step = record["step"]
            self.records.append(record)
            self.records = self.records[-max(2 * self.policy["stagnation_window"], 2) :]
            if self.reason:
                return
            if not all(math.isfinite(value) for value in record.values()):
                self._stop("Nonfinite electronic energy or residual")
                return
            ediff = self.incar.get("EDIFF", 1e-4)
            error = max(abs(record["de"]), abs(record["deps"]))
            if error <= ediff * 1.05:
                return  # Allow the next VASP convergence/termination message.
            limit = min(
                value
                for value in (self.policy["max_scf_steps"], self.incar.get("NELM", 60))
                if value > 0
            )
            if record["step"] >= limit:
                self._stop(
                    f"Electronic iteration limit {limit:g} reached without EDIFF convergence"
                )
                return
            window = self.policy["stagnation_window"]
            if (
                window
                and record["step"] >= self.policy["min_scf_steps"]
                and len(self.records) >= 2 * window
            ):
                errors = [
                    max(abs(row["de"]), abs(row["deps"]))
                    for row in self.records[-2 * window :]
                ]
                old, new = min(errors[:window]), min(errors[window:])
                if new > ediff * self.policy["unconverged_factor"] and new >= old * (
                    1 - self.policy["min_relative_improvement"]
                ):
                    self._stop(
                        f"SCF stagnation over {2 * window} steps: best energy error {old:.6g} -> {new:.6g} eV, EDIFF={ediff:.6g}"
                    )

    def _stop(self, reason):
        if self.reason:
            return
        self.reason = reason
        self.stopped_at = self.clock()
        # LABORT stops at the next electronic step, rather than waiting for a
        # nonconverging ionic step. These outputs must never become labels.
        (self.root / "STOPCAR").write_text("LABORT = .TRUE.\n")
        marker = {
            "reason": reason,
            "time": time.time(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "last_scf": {
                key: value if math.isfinite(value) else None
                for key, value in self.records[-1].items()
            }
            if self.records
            else None,
        }
        temporary = self.root / "fp-monitor-stop.tmp"
        temporary.write_text(json.dumps(marker, indent=2, allow_nan=False))
        os.replace(temporary, self.root / "fp-monitor-stop.json")
        print(f"FP monitor: stop requested: {reason}", flush=True)

    def tick(self):
        with self.lock:
            for name in ("OUTCAR", "OSZICAR"):
                try:
                    path = self.root / name
                    stat = path.stat()
                    current = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
                    previous = self.sizes[name]
                    if current != previous:
                        self.activity = self.clock()
                        self.sizes[name] = current
                        if name == "OSZICAR":
                            if (
                                previous is None
                                or current[0] != previous[0]
                                or current[1] < self.oszicar_position
                            ):
                                self.oszicar_position, self.partial = 0, b""
                            with path.open("rb") as stream:
                                stream.seek(self.oszicar_position)
                                data = stream.read(1048576)
                                self.oszicar_position = stream.tell()
                            lines = (self.partial + data).split(b"\n")
                            self.partial = lines.pop()
                            for line in lines:
                                self.feed(line.decode(errors="replace"))
                except OSError:
                    pass
            now = self.clock()
            if self.reason:
                elapsed = now - self.stopped_at
                sig = (
                    signal.SIGKILL
                    if elapsed
                    >= self.policy["stop_grace_period"]
                    + self.policy["kill_grace_period"]
                    else signal.SIGTERM
                    if elapsed >= self.policy["stop_grace_period"]
                    else None
                )
                if sig is not None and self.termination != sig:
                    try:
                        os.killpg(self.process.pid, sig)
                    except ProcessLookupError:
                        pass
                    self.termination = sig
                return
            if self.process is not None and self.process.poll() is not None:
                return
            if (
                self.policy["max_runtime"]
                and now - self.started >= self.policy["max_runtime"]
            ):
                self._stop("Configured single-point wall-time budget exceeded")
            elif (
                not self.last_step
                and self.policy["startup_timeout"]
                and now - self.started >= self.policy["startup_timeout"]
            ):
                self._stop("No electronic iteration before startup timeout")
            elif (
                self.last_step
                and self.policy["stall_timeout"]
                and now - self.activity >= self.policy["stall_timeout"]
            ):
                self._stop("No stdout, OSZICAR or OUTCAR progress before stall timeout")

    def _watch(self):
        while not self.done.wait(self.policy["check_interval"]):
            self.tick()

    def close(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join()
