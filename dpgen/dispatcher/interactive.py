"""Execute workflow tasks in the current GPU allocation without a scheduler."""

import os
import re
import shlex
import signal
import subprocess
import time
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path

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

    def _script(self, group):
        resources = self.resources
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
        lines.append(" && ".join(self.commands) + " || exit $?")
        lines.extend(resources.get("append_script", []))
        return "\n".join(lines)

    def run_submission(self):
        """Fill freed slots immediately, and return only after the stage succeeds."""
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
                        stdout = logs.enter_context(open(directory / self.outlog, "a"))
                        stderr = (
                            stdout
                            if self.errlog == self.outlog
                            else logs.enter_context(open(directory / self.errlog, "a"))
                        )
                        process = subprocess.Popen(
                            ["bash", "-c", self._script(group)],
                            cwd=directory,
                            stdin=subprocess.DEVNULL,
                            stdout=stdout,
                            stderr=stderr,
                            start_new_session=True,
                        )
                        active.append((process, group, task, stdout, stderr))
                    for item in active[:]:
                        process, group, task, stdout, stderr = item
                        code = process.poll()
                        if code is None:
                            continue
                        if code != 0:
                            raise RuntimeError(
                                f"Local task {self.work_path / task} failed with exit code {code}; "
                                f"see {self.outlog} and {self.errlog}"
                            )
                        active.remove(item)
                        stdout.close()
                        if stderr is not stdout:
                            stderr.close()
                        free.append(group)
                    if active:
                        time.sleep(0.05)
            finally:
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
