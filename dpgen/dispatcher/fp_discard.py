"""Durable user requests to discard and cancel selected recovery-managed FP tasks."""

import fcntl
import json
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from shlex import quote

from dpdispatcher.submission import JobStatus

from dpgen import dlog
from dpgen.dispatcher.fp_recovery_worker import atomic_json

REQUESTS = ".fp-discard-requests.json"


@contextmanager
def request_lock(root):
    with (Path(root) / ".fp-discard-requests.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def read_requests(root):
    path = Path(root) / REQUESTS
    return (
        json.loads(path.read_text())
        if path.exists()
        else {"version": 1, "requests": {}}
    )


def request_discard(root, jobs=(), tasks=(), reason="", dry_run=False):
    root = Path(root).resolve()
    if root.name != "02.fp" or not (root / ".fp-recovery.json").is_file():
        raise ValueError(
            "Select an existing 02.fp directory with a .fp-recovery.json record"
        )
    if not reason.strip() or not (jobs or tasks):
        raise ValueError("Provide --jobs or --tasks and a nonempty --reason")
    ledger = json.loads((root / ".fp-recovery.json").read_text())
    if ledger.get("stage", "FP") != "FP" or not re.fullmatch(
        r"[0-9a-f]{40}", ledger.get("submission_hash", "")
    ):
        raise ValueError(
            "FP campaign is not initialized; resume the updated recovery controller first"
        )
    selected = set(tasks)
    for job_id in jobs:
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", str(job_id)):
            raise ValueError(f"Invalid Slurm job ID: {job_id}")
        matches = [
            name
            for name, entry in ledger["tasks"].items()
            if str(entry.get("job_id")) == str(job_id)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Job {job_id} does not uniquely identify a current task in this stage"
            )
        selected.update(matches)
    for name in selected:
        if name not in ledger["tasks"] or Path(name).name != name:
            raise ValueError(f"Unknown FP task: {name}")
        if not (root / name).is_dir():
            raise ValueError(f"Missing FP task directory: {name}")
        if ledger["tasks"][name].get("status") == "successful":
            raise ValueError(
                f"{name} already has validated results; refusing to discard successful labels"
            )
    preview = [
        {
            "task": name,
            "job_id": ledger["tasks"][name].get("job_id"),
            "status": ledger["tasks"][name].get("status"),
        }
        for name in sorted(selected)
    ]
    if dry_run:
        return preview
    with request_lock(root):
        latest = json.loads((root / ".fp-recovery.json").read_text())
        if latest.get("submission_hash") != ledger.get("submission_hash"):
            raise ValueError(
                "FP submission changed during selection; select current tasks again"
            )
        for name in selected:
            current = latest.get("tasks", {}).get(name, {})
            if (
                current.get("input_digest") != ledger["tasks"][name].get("input_digest")
                or current.get("status") == "successful"
            ):
                raise ValueError(
                    f"{name} changed or completed during selection; no discard queued"
                )
        ledger = latest
        queue = read_requests(root)
        if queue.get("submission_hash", ledger.get("submission_hash")) != ledger.get(
            "submission_hash"
        ):
            raise ValueError("Discard queue belongs to another submission")
        queue["submission_hash"] = ledger.get("submission_hash")
        for name in sorted(selected):
            entry = ledger["tasks"][name]
            old = queue["requests"].get(name, {})
            request = (
                dict(old)
                if old.get("status") not in (None, "done", "error")
                else {
                    "id": uuid.uuid4().hex,
                    "requested_at": time.time(),
                    "status": "requested",
                }
            )
            request.update(
                task=name,
                reason=reason.strip(),
                selected_job_id=entry.get("job_id"),
                input_digest=entry.get("input_digest"),
                submission_hash=ledger.get("submission_hash"),
            )
            queue["requests"][name] = request
        atomic_json(root / REQUESTS, queue)
        # Postprocessing also reads scoped queue intents, so a completed job
        # cannot become a label while cancellation is waiting for the controller.
    return preview


def update_request(root, name, request_id, **changes):
    with request_lock(root):
        queue = read_requests(root)
        current = queue["requests"].get(name, {})
        if current.get("id") == request_id:
            current.update(changes)
            atomic_json(Path(root) / REQUESTS, queue)


def _text(stream):
    value = stream.read() if hasattr(stream, "read") else stream
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def apply_requests(submission):
    if submission.recovery_ledger.get("stage", "FP") != "FP":
        return
    root = submission.recovery_root
    if not (root / REQUESTS).exists():
        return
    with request_lock(root):
        requests = read_requests(root)["requests"]
    jobs = {
        job.job_task_list[0].task_work_path: job for job in submission.belonging_jobs
    }
    terminal = {
        JobStatus.finished,
        JobStatus.terminated,
        getattr(JobStatus, "failed", JobStatus.terminated),
    }
    for name, request in requests.items():
        if request.get("status") in ("done", "error"):
            continue
        entry = submission.recovery_ledger["tasks"].get(name)
        if (
            name not in jobs
            or len(jobs[name].job_task_list) != 1
            or entry is None
            or request.get("submission_hash") != submission.submission_hash
            or request.get("input_digest") != entry.get("input_digest")
        ):
            update_request(
                root,
                name,
                request["id"],
                status="error",
                error="Task, inputs or submission identity changed; no scheduler action taken",
            )
            dlog.error(
                "FP discard %s: identity mismatch; no scheduler action taken", name
            )
            continue
        job = jobs[name]
        action = entry.get("manual_discard", {})
        if not action or action.get("request_id") != request["id"]:
            action = {
                "request_id": request["id"],
                "reason": request["reason"],
                "requested_at": request["requested_at"],
            }
            entry["manual_discard"] = action
            entry.update(status="discard_pending", reason=request["reason"])
            atomic_json(root / name / "fp-discard.json", request)
            # Commit the no-retry intent before any scheduler cancellation.
            submission._save()
            with (root / "fp-recovery-history.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "task": name,
                            "job_id": str(job.job_id),
                            "category": "manual_discard",
                            "reason": request["reason"],
                            "request_id": request["id"],
                        }
                    )
                    + "\n"
                )
            dlog.warning(
                "FP discard %s job=%s: %s; retries disabled",
                name,
                job.job_id,
                request["reason"],
            )
        entry["status"] = "discard_pending"
        # Scheduler state was refreshed before this handler. No allocation needs
        # cancellation for a terminal job or a task that was never submitted.
        if job.job_state in terminal or (
            job.job_state == JobStatus.unsubmitted
            and not job.job_id
            and not entry.get("submission_in_progress")
        ):
            if job.job_id:
                submission._download_job(job)
            entry["status"] = "discarded"
            action["completed_at"] = time.time()
            submission._save()
            update_request(
                root,
                name,
                request["id"],
                status="done",
                completed_at=action["completed_at"],
                cancelled_job_id=action.get("cancelled_job_id"),
            )
            continue
        job_id = str(job.job_id) if job.job_id else ""
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
            dlog.warning(
                "FP discard %s: waiting for a verified scheduler job ID; no retry or cancellation attempted",
                name,
            )
            continue
        if action.get("cancelled_job_id") == job_id:
            continue  # Wait for scheduler confirmation; never declare it done early.
        try:
            code, _, stdout, stderr = submission.machine.context.block_call(
                "scancel " + quote(job_id)
            )
        except (OSError, RuntimeError) as exc:
            code, stdout, stderr = 1, "", str(exc)
        if code:
            error = (_text(stderr) or _text(stdout))[-4000:]
            action["cancel_error"] = error
            dlog.error(
                "FP discard %s job=%s: cancellation failed: %s; retry remains disabled",
                name,
                job_id,
                error,
            )
            submission._save()
            update_request(
                root, name, request["id"], status="cancel_pending", cancel_error=error
            )
            continue
        action.update(cancelled_job_id=job_id, cancel_requested_at=time.time())
        action.pop("cancel_error", None)
        submission._save()
        update_request(
            root, name, request["id"], status="cancel_pending", cancelled_job_id=job_id
        )
        dlog.info(
            "FP discard %s: cancellation requested for job %s; waiting for scheduler confirmation",
            name,
            job_id,
        )


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "fp-discard",
        help="Discard selected FP tasks and request cancellation without retry.",
    )
    parser.add_argument("STAGE", help="Existing iter.XXXXXX/02.fp directory")
    parser.add_argument(
        "--jobs", nargs="+", default=[], help="Current Slurm job IDs from this FP stage"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=[], help="Task directory names from this FP stage"
    )
    parser.add_argument(
        "--reason", required=True, help="Reason to preserve in the discard audit record"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview selected tasks without changing anything",
    )
    parser.set_defaults(func=command)


def command(args):
    preview = request_discard(
        args.STAGE, args.jobs, args.tasks, args.reason, args.dry_run
    )
    for item in preview:
        print(f"{item['task']}: job={item['job_id']} status={item['status']}")
    if args.dry_run:
        print("Preview only; no files changed and no cancellation requested.")
    else:
        print(
            "Discard requests queued. The updated recovery controller cancels these jobs at its next check; if it is stopped, resume it with the same DP-GEN command. Existing results and files are preserved."
        )
