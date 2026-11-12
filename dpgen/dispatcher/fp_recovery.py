"""Persistent Slurm FP recovery with evidence-based, unlimited hardware retries."""

import fcntl
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from shlex import quote

import dpdata
import numpy as np
from dpdispatcher import Job, Resources, Submission
from dpdispatcher.submission import JobStatus

from dpgen import dlog
from dpgen.dispatcher import fp_recovery_worker as worker
from dpgen.dispatcher import fp_vasp_monitor as vasp_monitor

_existing_request = ContextVar("fp_existing_directory", default=None)


def add_recovery_args(parser):
    parser.add_argument(
        "--fp-existing-dir",
        metavar="PATH",
        help="Resume FP jobs from this existing staging directory (or submission JSON); never create a replacement directory.",
    )


@contextmanager
def recovery_execution(args):
    path = getattr(args, "fp_existing_dir", None)
    if path and not os.path.isabs(os.path.expanduser(path)):
        raise ValueError(
            "--fp-existing-dir requires an absolute path on the configured FP machine"
        )
    token = _existing_request.set(
        {"path": os.path.expanduser(path), "stage": None} if path else None
    )
    try:
        yield
    finally:
        _existing_request.reset(token)


def requested_directory(root):
    request = _existing_request.get()
    if request is None:
        return None
    stage = str(Path(root).resolve())
    if request["stage"] is None:
        request["stage"] = stage
    return request["path"] if request["stage"] == stage else None


FAILED = getattr(JobStatus, "failed", JobStatus.terminated)
ACTIVE_STATES = tuple(
    getattr(JobStatus, name)
    for name in ("waiting", "running", "completing")
    if hasattr(JobStatus, name)
)


DEFAULTS = {
    "enabled": False,
    "max_nonhardware_retries": 2,
    "hardware_retry_delay": 60,
    "nonhardware_retry_delay": 60,
    "check_interval": 30,
}


def options(jdata):
    global_options = jdata.get("job_recovery", {})
    fp_options = jdata.get("fp_recovery", {})
    if not isinstance(global_options, dict) or not isinstance(fp_options, dict):
        raise ValueError("job_recovery and fp_recovery must be objects")
    supplied = dict(global_options, **fp_options)
    if not isinstance(supplied, dict) or supplied.keys() - DEFAULTS.keys():
        raise ValueError("Invalid fp_recovery options")
    result = dict(DEFAULTS, **supplied)
    if not isinstance(result["enabled"], bool):
        raise ValueError("fp_recovery.enabled must be boolean")
    for key in DEFAULTS.keys() - {"enabled"}:
        value = result[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Invalid fp_recovery.{key}")
    if int(result["max_nonhardware_retries"]) != result["max_nonhardware_retries"]:
        raise ValueError("max_nonhardware_retries must be an integer")
    if result["check_interval"] <= 0:
        raise ValueError("fp_recovery.check_interval must be positive")
    return result


def validate_vasp(task):
    """Validate convergence and parseable, finite single-point labels."""
    task = Path(task)
    if vasp_monitor.stopped_task(task):
        return (
            False,
            "Single point was stopped by the FP monitor; outputs are not labels",
        )
    try:
        content = (task / "OUTCAR").read_text(errors="replace")
        if "Elapsed time" not in content and "Elapse" not in content:
            return False, "OUTCAR has no normal termination footer"
        if "EDIFF is reached" not in content:
            return False, "OUTCAR has no electronic-convergence confirmation"
        reference = dpdata.System(str(task / "POSCAR"), fmt="vasp/poscar")
        result = dpdata.LabeledSystem(str(task / "OUTCAR"), fmt="vasp/outcar")
        if len(result) != 1 or result.get_natoms() != reference.get_natoms():
            return False, "Label frame/atom count does not match the single-point input"
        for name in ("coords", "cells", "energies", "forces"):
            if not np.isfinite(result.data[name]).all():
                return False, f"Nonfinite label {name}"
        if not np.allclose(
            result.data["cells"], reference.data["cells"], atol=1e-4, rtol=0
        ):
            return False, "Label cell differs from the single-point POSCAR"
        cell = reference.data["cells"][0]
        displacement = result.data["coords"][0] - reference.data["coords"][0]
        # VASP may wrap otherwise identical input positions across a periodic
        # boundary. Accept lattice translations, but never an actual relaxation.
        displacement -= np.rint(displacement @ np.linalg.inv(cell)) @ cell
        if not np.allclose(displacement, 0, atol=1e-4, rtol=0):
            return False, "Label geometry differs from the single-point POSCAR"
        before = dict(zip(reference.data["atom_names"], reference.data["atom_numbs"]))
        after = dict(zip(result.data["atom_names"], result.data["atom_numbs"]))
        if {k: v for k, v in before.items() if v} != {
            k: v for k, v in after.items() if v
        }:
            return False, "Label composition differs from POSCAR"
        return True, "Converged, finite single-point energy and forces"
    except (
        OSError,
        ValueError,
        RuntimeError,
        AssertionError,
        KeyError,
        IndexError,
    ) as exc:
        return False, f"Cannot validate VASP output: {exc}"


def _text(stream):
    value = stream.read() if hasattr(stream, "read") else stream
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


class RecoverySubmission(Submission):
    """Keep the dispatcher's identity/recovery lifecycle; replace blind retries."""

    def bind_machine(self, machine, *, bind_context=True):
        parent = super().bind_machine
        if "bind_context" in inspect.signature(parent).parameters:
            parent(machine, bind_context=bind_context)
        else:
            # Older DPDispatcher cannot suppress context binding. Reproduce the
            # core bind operations so inspecting an existing record never changes
            # its authenticated session or resets the selected staging directory.
            self.machine = machine
            for job in self.belonging_jobs:
                job.machine = machine
            self.submission_hash = self.get_hash()
            if machine is not None:
                if bind_context:
                    machine.context.bind_submission(self)
                self.local_root = machine.context.temp_local_root
        directory = getattr(self, "existing_remote_root", None)
        if machine is not None and bind_context and directory:
            machine.context.remote_root = directory
            machine.context.temp_remote_root = os.path.dirname(directory)
        return self

    @classmethod
    def deserialize(cls, submission_dict, machine=None, *, bind_context=True):
        """Restore records without newer vendor-only constructor/deserializer kwargs."""
        if machine is None:
            raise ValueError("FP recovery requires the already configured machine")
        restored = cls(
            work_base=submission_dict["work_base"],
            machine=None,
            resources=Resources.deserialize(submission_dict["resources"]),
            forward_common_files=submission_dict.get("forward_common_files", []),
            backward_common_files=submission_dict.get("backward_common_files", []),
            task_list=[],
        )
        if "_abs_work_base" in submission_dict:
            restored._abs_work_base = submission_dict["_abs_work_base"]
        restored.previous_submission_hash = submission_dict.get(
            "previous_submission_hash"
        )
        restored.belonging_jobs = [
            Job.deserialize(value) for value in submission_dict["belonging_jobs"]
        ]
        restored.belonging_tasks = [
            task for job in restored.belonging_jobs for task in job.job_task_list
        ]
        for job in restored.belonging_jobs:
            state = job.job_state
            if state is None:
                state = JobStatus.waiting if job.job_id else JobStatus.unsubmitted
            job.job_state = JobStatus(state)
            for task in job.job_task_list:
                task.task_state = JobStatus(
                    getattr(task, "task_state", None) or job.job_state
                )
        restored.bind_machine(machine, bind_context=bind_context)
        return restored

    def try_recover_from_json(self):
        # Vendor implementations may call a newer deserialize signature or open
        # another SSH connection. Keep recovery in this compatibility adapter.
        name = f"{self.submission_hash}.json"
        if not self.machine.context.check_file_exists(name):
            return
        record = json.loads(self.machine.context.read_file(name))
        restored = self.deserialize(record, machine=self.machine, bind_context=False)
        if restored.submission_hash != self.submission_hash:
            raise RuntimeError(
                "Existing submission identity is incompatible; refusing to create replacement FP jobs"
            )
        self.belonging_jobs = restored.belonging_jobs
        self.belonging_tasks = restored.belonging_tasks
        self.bind_machine(self.machine)
        dlog.info(
            "FP recovery: restored existing submission %s at %s",
            self.submission_hash,
            self.machine.context.remote_root,
        )

    def configure_recovery(
        self, root, config, ledger, validator=validate_vasp, instrumented=True
    ):
        self.recovery_label = ledger.get("stage", "FP") + " recovery"
        self.recovery_root = Path(root).resolve()
        self.recovery_options = config
        self.recovery_ledger = ledger
        self.recovery_validator = validator
        self.instrumented = instrumented
        self.health_path = self.recovery_root.parent.parent / ".dpgen-gpu-health.json"
        self.health = (
            json.loads(self.health_path.read_text())
            if self.health_path.exists()
            else {"devices": {}}
        )
        self.continue_on_failure = True
        # Reconsider exhausted jobs when a new classifier recognizes evidence
        # already saved by an older controller. Keep their scheduler identity.
        for name, entry in ledger["tasks"].items():
            diagnostic = entry.get("diagnosis", {})
            if (
                entry.get("status") == "successful"
                or diagnostic.get("category") != "unknown"
            ):
                continue
            category, reason = worker.classify(
                diagnostic.get("evidence", ""),
                diagnostic.get("scheduler", {}).get("state", ""),
            )
            if category not in worker.UNLIMITED_RETRY_CATEGORIES:
                continue
            diagnostic.update(category=category, reason=reason)
            entry.update(reason=reason, reclassified_at=time.time())
            entry["nonhardware_failures"] = max(
                0, entry.get("nonhardware_failures", 0) - 1
            )
            counter = (
                "hardware_failures"
                if category == "hardware"
                else "gpu_initialization_failures"
            )
            entry[counter] = entry.get(counter, 0) + 1
            if entry.get("status") == "failed":
                entry.update(
                    status="retry_wait",
                    retry_after=time.time() + config["hardware_retry_delay"],
                )
            with (self.recovery_root / "fp-recovery-history.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            diagnostic,
                            reclassified_at=entry["reclassified_at"],
                            previous_category="unknown",
                        )
                    )
                    + "\n"
                )
            dlog.warning(
                "%s %s: saved unknown failure reclassified as %s; retry=unlimited; existing job and outputs preserved",
                self.recovery_label,
                name,
                category,
            )

    def run_submission(
        self, *, clean=False, continue_on_failure=True, check_interval=30
    ):
        """Resume existing work and monitor siblings without generic blind retries."""
        if clean:
            raise ValueError(
                "FP recovery retains staging directories; clean must be False"
            )
        if not self.belonging_jobs:
            self.generate_jobs()
        recorded_hash = self.recovery_ledger.get("submission_hash")
        if recorded_hash and recorded_hash != self.submission_hash:
            raise RuntimeError(
                "Dispatcher identity changed; refusing to create a second FP submission"
            )
        try:
            self.try_recover_from_json()
            for job in self.belonging_jobs:
                entry = self.recovery_ledger["tasks"][
                    job.job_task_list[0].task_work_path
                ]
                if not job.job_id and entry.get("job_id"):
                    job.job_id = entry["job_id"]
                    job.job_state = JobStatus.waiting
                if job.job_state == FAILED and entry.get("status") != "failed":
                    job.job_state = JobStatus.terminated
                    for task in job.job_task_list:
                        task.task_state = JobStatus.terminated
            self._save()
            self._activity("checking existing scheduler jobs")
            self.update_submission_state()
            pending = [
                job
                for job in self.belonging_jobs
                if job.job_state == JobStatus.unsubmitted
                and self.recovery_ledger["tasks"][
                    job.job_task_list[0].task_work_path
                ].get("status")
                not in ("successful", "failed", "discarded")
            ]
            if pending:
                self._activity("staging inputs for unsubmitted tasks")
                tasks, jobs = self.belonging_tasks, self.belonging_jobs
                self.belonging_jobs = pending
                self.belonging_tasks = [
                    task for job in pending for task in job.job_task_list
                ]
                try:
                    self.upload_jobs()
                finally:
                    self.belonging_tasks, self.belonging_jobs = tasks, jobs
            while True:
                self._activity("handling job states and retry decisions")
                self.handle_unexpected_submission_state(continue_on_failure=True)
                self.submission_to_json()
                self._save()
                if all(
                    entry.get("status") in ("successful", "failed", "discarded")
                    for entry in self.recovery_ledger["tasks"].values()
                ):
                    break
                self._log_progress(check_interval)
                time.sleep(check_interval)
                self._activity("checking existing scheduler jobs")
                self.update_submission_state()
            failed = [
                name
                for name, entry in self.recovery_ledger["tasks"].items()
                if entry.get("status") == "failed"
            ]
            if failed:
                raise RuntimeError(
                    f"{len(failed)} {self.recovery_ledger.get('stage', 'FP')} tasks exhausted nonhardware retries. "
                    f"Completed results are preserved in {self.recovery_root}; "
                    "see fp-recovery-history.jsonl for reasons."
                )
            return self.serialize()
        finally:
            self._save()
            try:
                self.submission_to_json()
            except (OSError, RuntimeError):
                dlog.exception(
                    "Could not update remote FP state; local manifest retained"
                )

    def _activity(self, phase, task=None):
        self.recovery_ledger["activity"] = {
            "phase": phase,
            "task": task,
            "started": time.time(),
        }
        self._save()
        dlog.info("%s: %s%s", self.recovery_label, phase, f" ({task})" if task else "")

    def _log_progress(self, interval):
        counts = {}
        for entry in self.recovery_ledger["tasks"].values():
            state = entry.get("status", "pending")
            counts[state] = counts.get(state, 0) + 1
        self.recovery_ledger["counts"] = counts
        self.recovery_ledger["activity"] = {
            "phase": "waiting for next scheduler check",
            "started": time.time(),
        }
        self._save()
        dlog.info(
            "%s progress: successful=%d/%d running=%d waiting=%d retry_wait=%d pending=%d failed=%d discarded=%d discard_pending=%d; next check in %s s",
            self.recovery_label,
            counts.get("successful", 0),
            len(self.recovery_ledger["tasks"]),
            counts.get("running", 0) + counts.get("completing", 0),
            counts.get("waiting", 0),
            counts.get("retry_wait", 0),
            counts.get("pending", 0) + counts.get("submitted", 0),
            counts.get("failed", 0),
            counts.get("discarded", 0),
            counts.get("discard_pending", 0),
            interval,
        )

    def _save(self):
        self.recovery_ledger["submission_hash"] = self.submission_hash
        self.recovery_ledger["remote_root"] = str(self.machine.context.remote_root)
        worker.atomic_json(
            self.recovery_root / ".fp-recovery.json", self.recovery_ledger
        )

    def _read_remote(self, path):
        try:
            value = self.machine.context.read_file(str(path))
            return value.decode(errors="replace") if isinstance(value, bytes) else value
        except (OSError, FileNotFoundError):
            local = self.recovery_root / path
            return local.read_text(errors="replace") if local.exists() else ""

    def _download_job(self, job):
        self._activity("collecting outputs", job.job_task_list[0].task_work_path)
        # Narrow the download view without changing submission identity or
        # scheduler jobs. Missing failed-attempt artifacts are diagnostic only.
        tasks, jobs = self.belonging_tasks, self.belonging_jobs
        self.belonging_tasks, self.belonging_jobs = job.job_task_list, [job]
        try:
            self.machine.context.download(self, check_exists=True, mark_failure=False)

            if getattr(self, "fp_monitor_options", {}).get("enabled", False):
                for filename in ("fp-monitor-stop.json", "OSZICAR"):
                    for task in job.job_task_list:
                        try:
                            value = self.machine.context.read_file(
                                str(Path(task.task_work_path) / filename)
                            )
                            target = self.recovery_root / task.task_work_path / filename
                            target.write_text(
                                value.decode(errors="replace")
                                if isinstance(value, bytes)
                                else value
                            )
                        except OSError:
                            pass
        finally:
            self.belonging_tasks, self.belonging_jobs = tasks, jobs

    def _scheduler_diagnostics(self, job):
        job_id = str(job.job_id)
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
            return {"error": "Missing or unsupported scheduler job ID"}
        command = f"sacct -n -X -P -j {quote(job_id)} --format=JobIDRaw,State,ExitCode,NodeList,Partition,Reason"
        try:
            code, _, stdout, stderr = self.machine.context.block_call(command)
            text = _text(stdout)
            if code:
                return {"error": _text(stderr)[-4000:]}
            for line in text.splitlines():
                parts = line.split("|")
                if parts and parts[0] == job_id and len(parts) >= 6:
                    return dict(
                        zip(
                            (
                                "job_id",
                                "state",
                                "exit_code",
                                "node",
                                "partition",
                                "reason",
                            ),
                            parts[:6],
                        )
                    )
            return {"error": "No accounting row yet", "raw": text[-4000:]}
        except (OSError, RuntimeError) as exc:
            return {"error": str(exc)}

    def update_submission_state(self):
        for job in self.belonging_jobs:
            name = job.job_task_list[0].task_work_path
            entry = self.recovery_ledger["tasks"][name]
            if entry.get("status") in ("successful", "failed", "discarded"):
                continue
            marker = self._read_remote(f"{job.job_hash}_job_id").strip()
            if re.fullmatch(r"[0-9]+(?:_[0-9]+)?", marker) and (
                marker != str(job.job_id)
                or (
                    entry.get("submission_in_progress")
                    and marker != str(entry.get("job_id"))
                )
            ):
                job.job_id = marker
                job.job_state = JobStatus.waiting
                entry.update(job_id=marker, submission_in_progress=False)
                dlog.info(
                    "FP recovery %s: recovered existing Slurm job %s from submission marker",
                    name,
                    marker,
                )
            elif entry.get("submission_in_progress"):
                self._save()
                raise RuntimeError(
                    f"Submission outcome for {name} is uncertain; preserving state instead of submitting a possible duplicate"
                )
        super().update_submission_state()
        for job in self.belonging_jobs:
            if job.job_state != JobStatus.finished:
                continue
            task = job.job_task_list[0]
            name = task.task_work_path
            entry = self.recovery_ledger["tasks"][name]
            if entry.get("status") in ("successful", "discarded", "discard_pending"):
                continue
            self._download_job(job)
            self._activity("validating stage results", name)
            valid, reason = self.recovery_validator(self.recovery_root / name)
            if valid:
                entry.update(status="successful", job_id=str(job.job_id), reason=reason)
                dlog.info(
                    "%s %s: validated success, job=%s; results preserved",
                    self.recovery_label,
                    name,
                    job.job_id,
                )
            else:
                entry["validation_error"] = reason
                job.job_state = JobStatus.terminated
                task.task_state = JobStatus.terminated
            self._save()

    def _diagnose(self, job, name):
        scheduler = self._scheduler_diagnostics(job)
        raw = self._read_remote(Path(name) / "fp-attempt.json")
        try:
            attempt = json.loads(raw) if raw else {}
        except ValueError:
            attempt = {}
        # Never attribute the preceding job's attempt record to a new allocation.
        if attempt.get("job_id") and str(attempt["job_id"]) != str(job.job_id):
            attempt = {}
        log = self._read_remote(Path(name) / "fp-runtime.log")[-16000:]
        if self.instrumented and not attempt:
            log = ""
        elif not log and not self.instrumented:
            log = self._read_remote(
                Path(name) / getattr(self, "recovery_log_file", "fp.log")
            )[-16000:]
        try:
            reader = getattr(job, "get_last_error_message", None)
            backend = (reader() or "") if callable(reader) else ""
        except (OSError, RuntimeError):
            backend = ""
        current_evidence = log if self.instrumented else log + "\n" + backend
        category, reason = worker.classify(current_evidence, scheduler.get("state", ""))
        if attempt.get("category") in worker.UNLIMITED_RETRY_CATEGORIES + (
            "scientific_stop",
        ):
            category, reason = attempt["category"], attempt["reason"]
        if (
            getattr(self, "fp_monitor_options", {}).get("enabled", False)
            and category not in worker.UNLIMITED_RETRY_CATEGORIES
        ):
            try:
                stop = json.loads(
                    self._read_remote(Path(name) / "fp-monitor-stop.json") or "{}"
                )
                if stop.get("job_id") and str(stop["job_id"]) == str(job.job_id):
                    category, reason = "scientific_stop", stop["reason"]
            except (ValueError, KeyError):
                pass
        validation_error = self.recovery_ledger["tasks"][name].get("validation_error")
        if validation_error and category not in worker.UNLIMITED_RETRY_CATEGORIES + (
            "scientific_stop",
        ):
            category, reason = "application", validation_error
        return {
            "time": time.time(),
            "task": name,
            "job_id": str(job.job_id),
            "category": category,
            "reason": reason,
            "scheduler": scheduler,
            "attempt": attempt,
            "evidence": log[-16000:],
            "dispatcher_error": backend[-4000:],
        }

    def _refresh_worker_policy(self, job, name):
        if not self.instrumented:
            return
        path = self.recovery_root / name / "fp-recovery.json"
        policy = json.loads(path.read_text())
        policy["bad_gpu_uuids"] = list(self.health["devices"])
        worker.atomic_json(path, policy)
        # A worker may have consumed/moved old staging files. Restore the same
        # task's inputs into the same submission context before retrying.
        self.machine.context.write_file(
            str(Path(name) / "fp-recovery.json"), write_str=json.dumps(policy)
        )
        # Refresh instrumentation only before submitting a new attempt. Existing
        # staging directories may contain the older, broken CUDA UUID lookup.
        self.machine.context.write_file(
            str(Path(name) / "fp_recovery_worker.py"),
            write_str=Path(worker.__file__).read_text(),
        )
        if self.recovery_ledger.get("stage", "FP") == "FP":
            policy["fp_monitor"] = getattr(
                self, "fp_monitor_options", vasp_monitor.options()
            )
            self.machine.context.write_file(
                str(Path(name) / "fp-recovery.json"), write_str=json.dumps(policy)
            )
            if policy["fp_monitor"]["enabled"]:
                self.machine.context.write_file(
                    str(Path(name) / "fp_vasp_monitor.py"),
                    write_str=Path(vasp_monitor.__file__).read_text(),
                )

    def _clear_finished_tags(self, job):
        names = [f"{job.job_hash}_job_tag_finished"]
        names += [
            str(Path(task.task_work_path) / f"{task.task_hash}_task_tag_finished")
            for task in job.job_task_list
        ]
        code, _, _, stderr = self.machine.context.block_call(
            "rm -f -- " + " ".join(quote(name) for name in names)
        )
        if code:
            raise RuntimeError(
                f"Cannot clear stale FP success markers: {_text(stderr)}"
            )

    def _submit_with_intent(self, job, entry):
        entry["job_id"] = str(job.job_id) if job.job_id else None
        entry["submission_in_progress"] = True
        self._save()
        self.submission_to_json()
        job.submit_job()
        entry.update(
            submission_in_progress=False,
            status="submitted" if job.job_id else "pending",
            job_id=str(job.job_id) if job.job_id else None,
        )
        self.submission_to_json()
        self._save()

    def handle_unexpected_submission_state(self, *, continue_on_failure=False):
        from dpgen.dispatcher.fp_discard import apply_requests

        apply_requests(self)
        for job in self.belonging_jobs:
            task = job.job_task_list[0]
            name = task.task_work_path
            entry = self.recovery_ledger["tasks"][name]
            entry["job_id"] = str(job.job_id) if job.job_id else None
            if entry.get("status") in ("failed", "discarded", "discard_pending"):
                continue
            if entry.get("status") == "successful":
                job.job_state = JobStatus.finished
                task.task_state = JobStatus.finished
                continue
            if job.job_state == JobStatus.unsubmitted:
                self._refresh_worker_policy(job, name)
                self._submit_with_intent(job, entry)
                continue
            if job.job_state in ACTIVE_STATES:
                entry["status"] = JobStatus(job.job_state).name
                continue
            if job.job_state == JobStatus.unknown:
                self._save()
                raise RuntimeError(
                    f"Cannot establish scheduler state of {name}, job {job.job_id}; preserving state rather than submitting a duplicate"
                )
            if job.job_state != JobStatus.terminated:
                continue
            if entry.get("last_failure_job_id") != str(job.job_id):
                self._activity("diagnosing terminated Slurm job", name)
                diagnostic = self._diagnose(job, name)
                category = diagnostic["category"]
                entry.update(
                    last_failure_job_id=str(job.job_id),
                    diagnosis=diagnostic,
                    status="retry_wait",
                    reason=diagnostic["reason"],
                )
                counter = (
                    "hardware_failures"
                    if category == "hardware"
                    else "gpu_initialization_failures"
                    if category == "gpu_initialization"
                    else "scientific_stops"
                    if category == "scientific_stop"
                    else "nonhardware_failures"
                )
                entry[counter] = entry.get(counter, 0) + 1
                delay = self.recovery_options[
                    "hardware_retry_delay"
                    if category in worker.UNLIMITED_RETRY_CATEGORIES
                    else "nonhardware_retry_delay"
                ]
                entry["retry_after"] = time.time() + delay
                for uuid in (
                    diagnostic["attempt"].get("faulty_gpu_uuids", [])
                    if category == "hardware"
                    else []
                ):
                    self.health["devices"][uuid] = {
                        "node": diagnostic["attempt"].get("node"),
                        "partition": diagnostic["attempt"].get("partition"),
                        "evidence": diagnostic["reason"],
                        "time": time.time(),
                    }
                worker.atomic_json(self.health_path, self.health)
                with (self.recovery_root / "fp-recovery-history.jsonl").open(
                    "a"
                ) as stream:
                    stream.write(json.dumps(diagnostic) + "\n")
                dlog.warning(
                    "%s %s job=%s node=%s partition=%s: %s; reason=%s; retry=%s",
                    self.recovery_label,
                    name,
                    job.job_id,
                    diagnostic["scheduler"].get(
                        "node", diagnostic["attempt"].get("node")
                    ),
                    diagnostic["scheduler"].get(
                        "partition", diagnostic["attempt"].get("partition")
                    ),
                    category,
                    diagnostic["reason"],
                    "unlimited " + category
                    if category in worker.UNLIMITED_RETRY_CATEGORIES
                    else "no retry; exclude from labels"
                    if category == "scientific_stop"
                    else "bounded nonhardware",
                )
                try:
                    self._download_job(job)
                finally:
                    self.submission_to_json()
                    self._save()
            if entry["diagnosis"]["category"] == "scientific_stop":
                entry["status"] = "discarded"
                job.job_state = FAILED
                task.task_state = FAILED
                job.failure_reason = entry["reason"]
                self._save()
                continue
            unlimited = (
                entry["diagnosis"]["category"] in worker.UNLIMITED_RETRY_CATEGORIES
            )
            if (
                not unlimited
                and entry.get("nonhardware_failures", 0)
                > self.recovery_options["max_nonhardware_retries"]
            ):
                job.job_state = FAILED
                task.task_state = FAILED
                job.failure_reason = entry["reason"]
                entry["status"] = "failed"
                self.submission_to_json()
                self._save()
                continue
            if time.time() < entry["retry_after"]:
                continue
            self._clear_finished_tags(job)
            # Do not use the backend's finite generic retry policy here.
            job.fail_count += 1
            if not getattr(self, "preserve_remote_checkpoints", False):
                if hasattr(job, "_ensure_forward_files_on_retry"):
                    job._ensure_forward_files_on_retry()
                else:
                    self.upload_jobs()
            self._refresh_worker_policy(job, name)
            self._submit_with_intent(job, entry)
            entry.pop("validation_error", None)
        self._save()


def _resource_difference_fields(saved, current, prefix=""):
    differences = []
    for key in sorted(saved.keys() | current.keys()):
        field = f"{prefix}.{key}" if prefix else key
        if key not in saved or key not in current:
            differences.append(field)
        elif isinstance(saved[key], dict) and isinstance(current[key], dict):
            differences.extend(
                _resource_difference_fields(saved[key], current[key], field)
            )
        elif saved[key] != current[key]:
            differences.append(field)
    return differences


def _load_existing(probe, locator, root, paths, command):
    context = probe.machine.context
    locator = os.path.normpath(locator)
    original_root = context.remote_root
    if re.fullmatch(r"[0-9a-f]{40}\.json", os.path.basename(locator)):
        directory, record_name = os.path.dirname(locator), os.path.basename(locator)
    else:
        directory = locator
        record_name = os.path.basename(directory) + ".json"
        if not re.fullmatch(r"[0-9a-f]{40}\.json", record_name):
            # Only list inside the supplied directory. Never search elsewhere
            # and never create it when it is missing.
            context.remote_root = directory
            try:
                code, _, stdout, stderr = context.block_call(
                    "find . -maxdepth 1 -type f -name '*.json' -print"
                )
                candidates = [
                    os.path.basename(line)
                    for line in _text(stdout).splitlines()
                    if re.fullmatch(r"[0-9a-f]{40}\.json", os.path.basename(line))
                ]
                if code or len(candidates) != 1:
                    raise ValueError(
                        "Existing directory must contain one submission JSON; supply its full JSON path if there are several"
                    )
                record_name = candidates[0]
            finally:
                context.remote_root = original_root
    # Context path policies intentionally reject reads outside the bound staging
    # root. Rebind to the user-selected existing root before reading its record.
    context.remote_root = directory
    try:
        raw = context.read_file(record_name)
    except OSError as exc:
        raise FileNotFoundError(
            f"Cannot find existing FP submission {directory}/{record_name}; no replacement will be created"
        ) from exc
    record = json.loads(raw)
    submission = RecoverySubmission.deserialize(
        record, machine=probe.machine, bind_context=False
    )
    expected_hash = record_name[:-5]
    if submission.submission_hash != expected_hash:
        raise ValueError(
            "Existing submission identity is incompatible with the current machine settings; restore the original machine.json"
        )
    differences = _resource_difference_fields(
        submission.resources.serialize(), probe.resources.serialize()
    )
    submission.resource_difference_fields = differences
    if differences:
        # Resource values can contain credentials in envs/prepend_script. Report
        # field names only; the saved record remains authoritative for this stage.
        dlog.warning(
            "FP recovery: machine.json resource fields differ: %s. "
            "Using the existing submission's saved resources for these jobs and retries; "
            "current settings apply to new submissions.",
            ", ".join(differences),
        )
    if Path(context.temp_local_root, submission.work_base).resolve() != root:
        raise ValueError(
            f"Existing directory belongs to a different iteration/stage ({submission.work_base}), not {root}"
        )
    recovered_names = [task.task_work_path for task in submission.belonging_tasks]
    if not recovered_names or not set(recovered_names).issubset(paths):
        raise ValueError("Existing submission contains different FP tasks")
    if any(len(job.job_task_list) != 1 for job in submission.belonging_jobs):
        raise ValueError("Existing submission must use one FP task per Slurm job")
    wrapped = all(
        task.command.strip() == "python fp_recovery_worker.py"
        for task in submission.belonging_tasks
    )
    if not wrapped and any(
        task.command.strip() != command.strip() for task in submission.belonging_tasks
    ):
        raise ValueError("Existing FP command differs from machine.json")
    # Verify scientific inputs before any upload can overwrite an existing job.
    for name in recovered_names:
        for filename in ("POSCAR", "INCAR", "POTCAR", "KPOINTS"):
            local = root / name / filename
            if local.exists():
                remote = context.read_file(os.path.join(directory, name, filename))
                content = remote if isinstance(remote, bytes) else remote.encode()
                if (
                    hashlib.sha256(local.read_bytes()).digest()
                    != hashlib.sha256(content).digest()
                ):
                    raise ValueError(
                        f"Existing FP input differs: {name}/{filename}; refusing to overwrite or resubmit"
                    )
        if wrapped:
            policy = json.loads(
                context.read_file(os.path.join(directory, name, "fp-recovery.json"))
            )
            if policy["command"] != command:
                raise ValueError(
                    "Existing wrapped FP command differs from machine.json"
                )
    missing = set(paths) - set(recovered_names)
    if any(not validate_vasp(root / name)[0] for name in missing):
        raise ValueError(
            "Existing directory lacks unfinished local FP tasks; refusing to create another submission"
        )
    submission.existing_remote_root = directory
    submission.bind_machine(probe.machine)
    return submission, wrapped, directory


def _check_conflicting_campaign(submission, ledger):
    if (
        not ledger
        or not ledger.get("submission_hash")
        or ledger["submission_hash"] == submission.submission_hash
    ):
        return
    ids = sorted(
        {
            str(entry["job_id"])
            for entry in ledger["tasks"].values()
            if entry.get("job_id")
            and entry.get("status") not in ("successful", "failed", "discarded")
        }
    )
    if not ids:
        return
    if any(not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", value) for value in ids):
        raise RuntimeError(
            "Another local recovery campaign has unknown job IDs; refusing to abandon it"
        )
    code, _, stdout, stderr = submission.machine.context.block_call(
        "squeue -h -j " + quote(",".join(ids)) + " -o '%i|%T'"
    )
    active = _text(stdout).strip()
    error = _text(stderr)
    if active or (code and "Invalid job id" not in error):
        raise RuntimeError(
            "Another local FP submission may still be active; finish/cancel its jobs before adopting this directory"
        )


def _run_recovery(
    root, jdata, mdata, forward_files, backward_files, log_file="fp.log", common=None
):
    from dpgen.dispatcher.Dispatcher import make_submission

    config = options(jdata)
    monitor_config = vasp_monitor.options(jdata.get("fp_monitor"))
    if jdata["fp_style"] != "vasp" or mdata["fp_machine"].get("batch_type") != "Slurm":
        raise ValueError("FP recovery currently supports VASP single points with Slurm")
    work_base = os.fspath(root)
    root = Path(root).resolve()
    command = mdata["fp_command"]
    paths = sorted(path.name for path in root.glob("task.*") if path.is_dir())
    fingerprints = {name: worker.input_digest(root / name) for name in paths}
    signature = hashlib.sha256(
        json.dumps(
            {
                "inputs": fingerprints,
                "command": command,
                "machine": mdata["fp_machine"],
                "resources": mdata["fp_resources"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest = root / ".fp-recovery.json"
    ledger = json.loads(manifest.read_text()) if manifest.exists() else None
    explicit = requested_directory(root)
    locator = explicit or (ledger.get("existing_directory") if ledger else None)
    if (
        not locator
        and mdata["fp_resources"].get("group_size", mdata.get("fp_group_size", 1)) != 1
    ):
        raise ValueError("FP recovery requires fp.resources.group_size=1")
    if ledger and ledger["signature"] != signature and not explicit:
        raise RuntimeError(
            "FP inputs or submission settings changed; refusing to overwrite or duplicate an existing recovery campaign"
        )
    builder = dict(
        commands=[command],
        work_path=work_base,
        run_tasks=paths,
        group_size=1,
        forward_common_files=list(common or []),
        forward_files=list(forward_files),
        backward_files=list(backward_files),
        outlog=log_file,
        errlog=log_file,
        submission_class=RecoverySubmission,
    )
    # Probe the original unwrapped identity first. Existing jobs keep their
    # original commands, IDs and staging root; instrumentation begins next stage.
    legacy = make_submission(mdata["fp_machine"], mdata["fp_resources"], **builder)
    if not isinstance(legacy, RecoverySubmission):
        raise ValueError("FP recovery requires batch Slurm execution without --gpus")
    legacy.generate_jobs()
    if locator:
        selected, wrapped, directory = _load_existing(
            legacy, locator, root, paths, command
        )
        _check_conflicting_campaign(selected, ledger)
        if ledger and (
            ledger.get("submission_hash") != selected.submission_hash
            or ledger["signature"] != signature
        ):
            shutil.copyfile(
                manifest, root / f".fp-recovery.before-adopt.{time.time_ns()}.json"
            )
            ledger = None
        if ledger is None:
            ledger = {
                "version": 1,
                "signature": signature,
                "instrumented": wrapped,
                "tasks": {
                    name: {"status": "pending", "input_digest": fingerprints[name]}
                    for name in paths
                },
            }
            for name in paths:
                valid, reason = validate_vasp(root / name)
                if valid:
                    ledger["tasks"][name].update(status="successful", reason=reason)
        ledger.update(
            existing_directory=directory,
            submission_hash=selected.submission_hash,
            remote_root=directory,
            instrumented=wrapped,
            resource_difference_fields=selected.resource_difference_fields,
        )
        worker.atomic_json(manifest, ledger)
        selected.configure_recovery(root, config, ledger, instrumented=wrapped)
        selected.fp_monitor_options = monitor_config
        dlog.info(
            "FP recovery: explicitly resuming %s; no replacement directory will be created",
            directory,
        )
        return selected.run_submission(
            clean=False,
            continue_on_failure=True,
            check_interval=config["check_interval"],
        )
    old_exists = legacy.machine.context.check_file_exists(
        f"{legacy.submission_hash}.json"
    )
    instrumented = ledger.get("instrumented", True) if ledger else not old_exists
    if ledger is None:
        ledger = {
            "version": 1,
            "signature": signature,
            "instrumented": instrumented,
            "tasks": {
                name: {"status": "pending", "input_digest": fingerprints[name]}
                for name in paths
            },
        }
        for name in paths:
            valid, reason = validate_vasp(root / name)
            if valid:
                ledger["tasks"][name].update(status="successful", reason=reason)
        worker.atomic_json(manifest, ledger)
    if all(entry["status"] == "successful" for entry in ledger["tasks"].values()):
        dlog.info(
            "FP recovery: all %d existing results validated; no jobs submitted",
            len(paths),
        )
        return
    if instrumented:
        if "campaign_tasks" not in ledger:
            ledger["campaign_tasks"] = [
                name
                for name in paths
                if ledger["tasks"][name]["status"] != "successful"
            ]
            worker.atomic_json(manifest, ledger)
        for name in ledger["campaign_tasks"]:
            task = root / name
            shutil.copyfile(worker.__file__, task / "fp_recovery_worker.py")
            existing = task / "fp-recovery.json"
            policy = (
                json.loads(existing.read_text())
                if existing.exists()
                else {"command": command, "bad_gpu_uuids": []}
            )
            policy["validated_input_digest"] = (
                fingerprints[name]
                if ledger["tasks"][name]["status"] == "successful"
                else None
            )
            policy["fp_monitor"] = monitor_config
            worker.atomic_json(existing, policy)
            if monitor_config["enabled"]:
                shutil.copyfile(vasp_monitor.__file__, task / "fp_vasp_monitor.py")
        builder.update(
            commands=["python fp_recovery_worker.py"],
            run_tasks=ledger["campaign_tasks"],
            forward_files=list(forward_files)
            + ["fp_recovery_worker.py", "fp-recovery.json"],
            backward_files=list(backward_files) + ["fp-attempt.json", "fp-runtime.log"],
        )
        submission = make_submission(
            mdata["fp_machine"], mdata["fp_resources"], **builder
        )
    else:
        submission = legacy
        dlog.info(
            "FP recovery: adopting existing submission %s; old jobs have no GPU wrapper identity",
            submission.submission_hash,
        )
    submission.configure_recovery(root, config, ledger, instrumented=instrumented)
    submission.fp_monitor_options = monitor_config
    if monitor_config["enabled"] and not instrumented:
        dlog.warning(
            "FP monitor: adopted uninstrumented jobs cannot be watched by the compute-node monitor; monitoring starts with new instrumented stages"
        )
    # Parent recovery runs before any submission. It keeps the same hash and
    # remote root, continues siblings, and never cleans away partial results.
    submission.run_submission(
        clean=False, continue_on_failure=True, check_interval=config["check_interval"]
    )


def run_recovery(
    root, jdata, mdata, forward_files, backward_files, log_file="fp.log", common=None
):
    """Hold an exclusive campaign lock to prevent duplicate controller submits."""
    with (Path(root) / ".fp-recovery.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another controller is already managing this FP stage"
            ) from exc
        return _run_recovery(
            root, jdata, mdata, forward_files, backward_files, log_file, common
        )
