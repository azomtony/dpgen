"""Allocation worker: runs assigned jobs side by side on one allocation's GPUs.

Jobs are FP single points (``fp_recovery_worker.py`` by default) or any command
given as ``argv`` in the assignment, such as a dpdispatcher job script for
training or exploration.

Copied into ``02.fp/allocations/<id>/`` and started by that allocation's
``alloc.sbatch``. It needs only the Python standard library. The controller
talks to it through files in the allocation directory:

``inbox/<task>.json``      assignment written by the controller
``running/<task>.json``    the worker started it (GPUs, start time)
``finished/<task>.json``   it ended: exit code, or interrupted / not started
``cancel/<task>.json``     controller asks to stop that task
``drain``                  controller asks the worker to exit when idle
``status.json``            heartbeat: GPUs, busy slots, deadline

Each task runs ``fp_recovery_worker.py`` inside its own task directory with
``CUDA_VISIBLE_DEVICES`` limited to the GPUs given to it, so tasks never share
a GPU. On a stop signal (preemption, or the wall-time warning) running tasks are
stopped and reported as interrupted, so the controller requeues them.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

KILL_GRACE = 20


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def visible_gpus(environment, count):
    """GPU tokens this allocation may use, as Slurm exposed them."""
    tokens = [
        token.strip()
        for token in environment.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if token.strip() and token.strip() != "-1"
    ]
    if not tokens:
        tokens = [str(index) for index in range(count)]
    return tokens[:count] if count else tokens


class Pilot:
    def __init__(
        self,
        root,
        gpus,
        time_limit,
        idle_exit,
        python=sys.executable,
        clock=time.time,
        sleep=time.sleep,
        popen=subprocess.Popen,
        environment=None,
        killpg=os.killpg,
    ):
        self.root = Path(root).resolve()
        self.environment = dict(os.environ if environment is None else environment)
        self.clock = clock
        self.sleep = sleep
        self.popen = popen
        self.killpg = killpg
        self.python = python
        self.started = clock()
        self.deadline = self.started + float(time_limit)
        self.idle_exit = float(idle_exit)
        self.slots = visible_gpus(self.environment, int(gpus))
        self.job_id = self.environment.get("SLURM_JOB_ID")
        self.node = self.environment.get("SLURMD_NODENAME") or socket.gethostname()
        self.busy = {}
        self.last_activity = self.started
        self.signal = None
        self.exit_reason = None
        for name in ("inbox", "running", "finished", "cancel"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    # ---- bookkeeping -----------------------------------------------------

    def free(self):
        used = {gpu for info in self.busy.values() for gpu in info["gpus"]}
        return [gpu for gpu in self.slots if gpu not in used]

    def heartbeat(self, state="running"):
        atomic_json(
            self.root / "status.json",
            {
                "state": state,
                "job_id": self.job_id,
                "node": self.node,
                "pid": os.getpid(),
                "started": self.started,
                "deadline": self.deadline,
                "time": self.clock(),
                "gpus": self.slots,
                "busy": {name: info["gpus"] for name, info in self.busy.items()},
                "exit_reason": self.exit_reason,
            },
        )

    def _finish(self, name, **fields):
        record = {"task": name, "job_id": self.job_id, "node": self.node, **fields}
        atomic_json(self.root / "finished" / f"{name}.json", record)
        try:
            (self.root / "running" / f"{name}.json").unlink()
        except FileNotFoundError:
            pass

    # ---- task lifecycle --------------------------------------------------

    def poll(self):
        for name, info in list(self.busy.items()):
            code = info["process"].poll()
            if code is None:
                continue
            info["log"].close()
            self._finish(
                name,
                exit_code=code,
                start=info["start"],
                end=self.clock(),
                gpus=info["gpus"],
                attempt=info["assignment"].get("attempt"),
            )
            del self.busy[name]
            self.last_activity = self.clock()

    def _stop_task(self, name, **fields):
        info = self.busy.pop(name)
        process = info["process"]
        for sig, wait in ((signal.SIGTERM, KILL_GRACE), (signal.SIGKILL, 5)):
            if process.poll() is not None:
                break
            try:
                self.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            deadline = self.clock() + wait
            while process.poll() is None and self.clock() < deadline:
                self.sleep(0.5)
        info["log"].close()
        self._finish(
            name,
            start=info["start"],
            end=self.clock(),
            gpus=info["gpus"],
            attempt=info["assignment"].get("attempt"),
            exit_code=process.poll(),
            **fields,
        )
        self.last_activity = self.clock()

    def handle_cancels(self):
        for path in sorted((self.root / "cancel").glob("*.json")):
            name = path.stem
            if name in self.busy:
                self._stop_task(name, cancelled=True)
            inbox = self.root / "inbox" / f"{name}.json"
            if inbox.exists():
                inbox.unlink()
                self._finish(name, cancelled=True, not_started=True)
            path.unlink()

    def start_tasks(self):
        waiting = []
        for path in (self.root / "inbox").glob("*.json"):
            try:
                waiting.append((path, json.loads(path.read_text())))
            except ValueError:
                continue  # still being written
        # Widest and longest first, so multi-GPU tasks are not starved by
        # single-GPU ones taking scattered GPUs.
        waiting.sort(
            key=lambda item: (
                -int(item[1].get("ngpus", 1)),
                -float(item[1].get("estimate", 0)),
                item[0].name,
            )
        )
        for path, assignment in waiting:
            name = assignment["name"]
            need = int(assignment.get("ngpus", 1))
            if need > len(self.slots):
                path.unlink()
                self._finish(
                    name,
                    not_started=True,
                    reason=f"needs {need} GPUs; this allocation has {len(self.slots)}",
                )
                continue
            remaining = self.deadline - self.clock()
            if float(assignment.get("estimate", 0)) > remaining:
                path.unlink()
                self._finish(
                    name,
                    not_started=True,
                    reason=f"not enough time left ({remaining / 3600:.1f} h)",
                )
                continue
            free = self.free()
            if need > len(free):
                continue
            gpus = free[:need]
            task = Path(assignment["task_dir"])
            environment = dict(self.environment)
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
            environment["DPGEN_ALLOCATION"] = str(self.root)
            log_name = assignment.get("log") or f"fp-pilot-{self.job_id or 'local'}.log"
            log = open(task / log_name, "a")
            argv = assignment.get("argv") or [self.python, "fp_recovery_worker.py"]
            process = self.popen(
                list(argv),
                cwd=str(task),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            start = self.clock()
            self.busy[name] = {
                "process": process,
                "gpus": gpus,
                "start": start,
                "log": log,
                "assignment": assignment,
            }
            atomic_json(
                self.root / "running" / f"{name}.json",
                {**assignment, "gpus": gpus, "start": start, "node": self.node},
            )
            path.unlink()
            self.last_activity = start

    def stop_all(self, signum):
        for name in list(self.busy):
            self._stop_task(name, interrupted=True, signal=int(signum))

    def idle(self):
        return not self.busy and not any((self.root / "inbox").glob("*.json"))

    # ---- main loop -------------------------------------------------------

    def _on_signal(self, signum, frame):
        self.signal = signum

    def run(self, interval=5.0, max_loops=None):
        for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
            try:
                signal.signal(sig, self._on_signal)
            except ValueError:
                pass  # not the main thread (tests)
        loops = 0
        while True:
            self.heartbeat()
            self.handle_cancels()
            self.poll()
            if self.signal is not None:
                self.stop_all(self.signal)
                self.exit_reason = f"signal {int(self.signal)}"
                break
            self.start_tasks()
            if self.idle():
                if (self.root / "drain").exists():
                    self.exit_reason = "drained by controller"
                    break
                if self.clock() - self.last_activity >= self.idle_exit:
                    self.exit_reason = "idle"
                    break
            loops += 1
            if max_loops is not None and loops >= max_loops:
                break
            self.sleep(interval)
        self.heartbeat(state="exited")
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--time-limit", type=float, required=True, help="seconds")
    parser.add_argument("--idle-exit", type=float, default=900)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    pilot = Pilot(Path.cwd(), args.gpus, args.time_limit, args.idle_exit, args.python)
    return pilot.run()


if __name__ == "__main__":
    sys.exit(main())
