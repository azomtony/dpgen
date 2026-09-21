"""Execute workflow tasks in the current GPU allocation without a scheduler."""

import hashlib
import os
import re
import shlex
import signal
import subprocess
import time
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path

from dpgen import dlog

_gpu_groups = ContextVar("dpgen_gpu_groups", default=None)


def add_interactive_args(parser):
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        help="Run locally on these CUDA device IDs instead of submitting batch jobs.",
    )
    parser.add_argument(
        "--gpus-per-job",
        type=int,
        default=None,
        help="GPUs assigned to each local task (default: 1).",
    )


@contextmanager
def interactive_execution(args):
    """Scope local execution to this workflow invocation."""
    gpus = getattr(args, "gpus", None)
    size = getattr(args, "gpus_per_job", None)
    groups = None
    if gpus is None:
        if size is not None:
            raise ValueError("--gpus-per-job requires --gpus")
    else:
        size = 1 if size is None else size
        if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
            raise ValueError("--gpus requires distinct non-negative device IDs")
        if size < 1 or size > len(gpus) or len(gpus) % size:
            raise ValueError("--gpus-per-job must be positive and divide the GPU count")
        groups = [gpus[i : i + size] for i in range(0, len(gpus), size)]
    token = _gpu_groups.set(groups)
    try:
        yield
    finally:
        _gpu_groups.reset(token)


def local_submission(resources, commands, work_path, run_tasks, outlog, errlog):
    groups = _gpu_groups.get()
    if groups is None:
        return None
    return LocalSubmission(
        groups, resources, commands, work_path, run_tasks, outlog, errlog
    )


class LocalSubmission:
    """Blocking stage executor with exclusive GPU slots and process cleanup."""

    def __init__(self, groups, resources, commands, work_path, tasks, outlog, errlog):
        self.groups = groups
        self.resources = resources
        self.commands = commands
        self.work_path = Path(work_path).resolve()
        self.tasks = list(tasks)
        self.outlog = outlog
        self.errlog = errlog

    def _stage_label(self):
        stage = {
            "00.train": "training",
            "01.model_devi": "exploration",
            "02.fp": "labeling",
        }.get(self.work_path.name, self.work_path.name)
        iteration = next(
            (
                p.name
                for p in (self.work_path, *self.work_path.parents)
                if re.fullmatch(r"iter\.\d+", p.name)
            ),
            "iteration unknown",
        )
        return f"{iteration} | {stage}"

    def _task_label(self, task, group):
        parts = [self._stage_label()]
        if self.work_path.name == "00.train" and str(task).isdigit():
            parts.append(f"model {task}")
        else:
            match = re.fullmatch(r"task\.(\d+)\.(\d+)", str(task))
            if match:
                parts.append(f"system {match[1]} | task {match[2]}")
            else:
                parts.append(f"task {task}")
        parts.append("GPU " + ",".join(map(str, group)))
        return " | ".join(parts)

    def _steps(self):
        steps = list(self.commands)
        if self.resources.get("append_script"):
            steps.append("\n".join(self.resources["append_script"]))
        return steps

    def _resume_state(self, directory):
        """Match successful command prefixes against the current input and commands."""
        digest = hashlib.sha256()
        input_file = directory / "input.json"
        if input_file.is_file():
            digest.update(input_file.read_bytes())
        keys = []
        for command in self._steps():
            digest.update(b"\0" + command.encode())
            keys.append(digest.hexdigest())
        resume = 0
        for index, key in enumerate(keys):
            marker = directory / ".dpgen-local-progress" / f"{index:03d}.done"
            if not marker.is_file() or marker.read_text().strip() != key:
                break
            resume += 1
        return resume, keys

    def _environment_lines(self, group, resources):
        lines = ["set -e"]
        lines.extend(resources.get("prepend_script", []))
        if resources.get("module_purge", False):
            lines.append("module purge")
        lines.extend(
            "module unload " + shlex.quote(module)
            for module in resources.get("module_unload_list", [])
        )
        lines.extend(
            "module load " + shlex.quote(module)
            for module in resources.get("module_list", [])
        )
        lines.extend(
            "source " + shlex.quote(source)
            for source in resources.get("source_list", [])
        )
        for key, value in resources.get("envs", {}).items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid environment variable name: {key}")
            lines.append(f"export {key}={shlex.quote(str(value))}")
        # Set this last so stage setup cannot overwrite the task's allocation.
        lines.append("export CUDA_VISIBLE_DEVICES=" + ",".join(map(str, group)))
        return lines

    def _script(self, group, directory=None, resume=0, keys=None):
        resources = self.resources
        if directory is None:
            lines = self._environment_lines(group, resources)
            lines.append(" && ".join(self.commands) + " || exit $?")
            lines.extend(resources.get("append_script", []))
        else:
            lines = ["set -e"]
            progress = directory / ".dpgen-local-progress"
            progress.mkdir(exist_ok=True)
            # Discard stale downstream successes before restarting an earlier step.
            for marker in progress.glob("*.done"):
                if marker.stem.isdigit() and int(marker.stem) >= resume:
                    marker.unlink()
            for index, command in enumerate(self._steps()[resume:], start=resume):
                marker = shlex.quote(str(progress / f"{index:03d}.done"))
                temporary = shlex.quote(str(progress / f"{index:03d}.tmp"))
                lines.append(f"echo 'DP-GEN command {index + 1}: started'")
                name = "command"
                if index < len(self.commands):
                    name = {0: "train", 1: "freeze", 2: "compress"}.get(
                        index, "command"
                    )
                override = resources.get("command_envs", {}).get(name, {})
                command_resources = {**resources, **override}
                command_resources["envs"] = {
                    **resources.get("envs", {}),
                    **override.get("envs", {}),
                }
                setup = self._environment_lines(group, command_resources)
                setup.append(command)
                # A fresh shell prevents the training environment leaking into freeze.
                lines.append("bash -c " + shlex.quote("\n".join(setup)) + " || exit $?")
                lines.append(f"printf '%s\\n' {shlex.quote(keys[index])} > {temporary}")
                lines.append(f"mv {temporary} {marker}")
                lines.append(f"echo 'DP-GEN command {index + 1}: completed'")
        return "\n".join(lines)

    def run_submission(self):
        """Fill freed slots immediately, and return only after the stage succeeds."""
        stage = self._stage_label()
        total = len(self.tasks)
        completed = 0
        started = {}
        stage_start = time.monotonic()
        last_progress = stage_start
        dlog.info(
            "%s | starting %d tasks | %d GPU slots", stage, total, len(self.groups)
        )
        if self.work_path.name == "01.model_devi":
            models = sorted(
                p.name
                for p in self.work_path.glob("graph*")
                if p.suffix in {".pb", ".pt", ".pth"}
            )
            if models:
                dlog.info("%s | model ensemble: %s", stage, ", ".join(models))
        pending = iter(self.tasks)
        free = list(self.groups)
        active = []
        exhausted = False
        with ExitStack() as logs:
            try:
                while active or not exhausted:
                    while free and not exhausted:
                        task = next(pending, None)
                        if task is None:
                            exhausted = True
                            break
                        group = free.pop(0)
                        directory = self.work_path / task
                        checkpoint_directory = (
                            directory if self.work_path.name == "00.train" else None
                        )
                        resume, keys = (
                            self._resume_state(directory)
                            if checkpoint_directory
                            else (0, [None] * len(self._steps()))
                        )
                        if resume == len(keys):
                            completed += 1
                            free.append(group)
                            dlog.info(
                                "%s | already completed; skipped | %d/%d complete",
                                self._task_label(task, group),
                                completed,
                                total,
                            )
                            continue
                        if resume:
                            dlog.info(
                                "%s | skipping %d completed commands; resuming command %d: %s",
                                self._task_label(task, group),
                                resume,
                                resume + 1,
                                self._steps()[resume],
                            )
                        stdout = logs.enter_context(open(directory / self.outlog, "a"))
                        stderr = (
                            stdout
                            if self.errlog == self.outlog
                            else logs.enter_context(open(directory / self.errlog, "a"))
                        )
                        process = subprocess.Popen(
                            [
                                "bash",
                                "-c",
                                self._script(group, checkpoint_directory, resume, keys),
                            ],
                            cwd=directory,
                            stdin=subprocess.DEVNULL,
                            stdout=stdout,
                            stderr=stderr,
                            start_new_session=True,
                        )
                        active.append((process, group, task, stdout, stderr))
                        started[task] = time.monotonic()
                        dlog.info(
                            "%s | started | log: %s",
                            self._task_label(task, group),
                            directory / self.outlog,
                        )
                    for item in active[:]:
                        process, group, task, stdout, stderr = item
                        code = process.poll()
                        if code is None:
                            continue
                        if code != 0:
                            dlog.error(
                                "%s | failed (exit %d) | logs: %s, %s",
                                self._task_label(task, group),
                                code,
                                self.work_path / task / self.outlog,
                                self.work_path / task / self.errlog,
                            )
                            raise RuntimeError(
                                f"Local task {self.work_path / task} failed with exit code {code}; "
                                f"see {self.outlog} and {self.errlog}"
                            )
                        completed += 1
                        dlog.info(
                            "%s | completed in %.1fs | %d/%d complete",
                            self._task_label(task, group),
                            time.monotonic() - started.pop(task),
                            completed,
                            total,
                        )
                        active.remove(item)
                        stdout.close()
                        if stderr is not stdout:
                            stderr.close()
                        free.append(group)
                    if active:
                        now = time.monotonic()
                        if now - last_progress >= 60:
                            for _, group, task, *_ in active:
                                dlog.info(
                                    "%s | running for %.1fs | %d/%d complete",
                                    self._task_label(task, group),
                                    now - started[task],
                                    completed,
                                    total,
                                )
                            last_progress = now
                        time.sleep(0.05)
                dlog.info(
                    "%s | completed %d/%d tasks in %.1fs",
                    stage,
                    completed,
                    total,
                    time.monotonic() - stage_start,
                )
            finally:
                if active:
                    dlog.warning(
                        "%s | stopping remaining tasks; stage incomplete", stage
                    )
                # Kill the whole process group, including MPI launcher children.
                for process, *_ in active:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                deadline = time.monotonic() + 2
                while (
                    any(p.poll() is None for p, *_ in active)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.05)
                for process, *_ in active:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
