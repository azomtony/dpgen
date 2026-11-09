"""Hardware-aware Slurm recovery for training and exploration campaigns."""

import fcntl
import hashlib
import json
import shutil
from pathlib import Path
from shlex import quote

from dpgen import dlog
from dpgen.dispatcher import fp_recovery_worker as worker
from dpgen.dispatcher.fp_recovery import RecoverySubmission, options


class StageSubmission(RecoverySubmission):
    def run_submission(self, **kwargs):
        # This lock covers polling as well as submission, so two controllers
        # cannot both retry the same task.
        with (self.recovery_root / ".fp-recovery.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another controller is managing this stage") from exc
            return super().run_submission(
                clean=False,
                continue_on_failure=True,
                check_interval=self.recovery_options["check_interval"],
            )


def validate_result(task, fingerprint, artifacts, plasma=False, require_receipt=True):
    """Require a successful instrumented attempt and its expected outputs.

    Scientific parsing remains in the normal stage postprocessing. A stale
    output file alone can never turn an interrupted attempt into success.
    """
    task = Path(task)
    try:
        if require_receipt:
            report = json.loads((task / "fp-attempt.json").read_text())
            if report.get("input_digest") != fingerprint:
                return False, "Attempt does not match the campaign inputs"
            if report.get("status") != "exited" or report.get("exit_code") != 0:
                return False, "Application has not completed successfully"
        for pattern in artifacts:
            if not any(p.is_file() and p.stat().st_size for p in task.glob(pattern)):
                return False, f"Missing or empty stage output: {pattern}"
        if plasma:
            state = json.loads((task / "plasma.status.json").read_text())
            if state.get("status") not in ("complete", "trust_stop"):
                return (
                    False,
                    "Plasma segment is not complete or stopped by trust policy",
                )
        return True, "Successful command chain and expected stage outputs"
    except (OSError, ValueError, TypeError) as exc:
        return False, f"Cannot validate stage output: {exc}"


def make_stage_submission(
    jdata, stage, machine, resources, *, ordinary_factory=None, **builder
):
    """Opt in via job_recovery; keep the ordinary dispatcher otherwise.

    New campaigns use one task per job so failures and completed siblings are
    independent, even when the ordinary stage groups multiple tasks together.
    """
    from dpgen.dispatcher.Dispatcher import make_submission

    supplied = jdata.get("job_recovery", {})
    if not isinstance(supplied, dict):
        raise ValueError("job_recovery must be an object")
    config = options({"fp_recovery": supplied})
    if not config["enabled"]:
        return (ordinary_factory or make_submission)(machine, resources, **builder)
    if machine.get("batch_type") != "Slurm":
        raise ValueError("job_recovery requires batch Slurm execution")
    if stage == "model_devi" and jdata.get("model_devi_engine", "lammps") != "lammps":
        raise ValueError("Exploration recovery currently supports LAMMPS")
    original_builder = dict(builder)
    # Training gathers common data through a set. Canonical ordering keeps the
    # dispatcher identity stable when the controller restarts in another process.
    builder = dict(builder)
    for key in ("run_tasks", "forward_common_files", "forward_files", "backward_files"):
        builder[key] = sorted(builder[key])
    root = Path(builder["work_path"]).resolve()
    paths = list(builder["run_tasks"])
    command = "&&".join(builder["commands"])
    # Dynamic state is deliberately excluded. Retrying must retain the newer
    # checkpoint in the persistent staging directory.
    dynamic = {"checkpoints", "traj"}
    input_files = [f for f in builder["forward_files"] if f not in dynamic]
    fingerprints = {
        name: worker.input_digest(root / name, input_files) for name in paths
    }
    signature = hashlib.sha256(
        json.dumps(
            {
                "stage": stage,
                "inputs": fingerprints,
                "command": command,
                "machine": machine,
                "resources": resources,
                "builder": builder,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest = root / ".fp-recovery.json"
    ledger = json.loads(manifest.read_text()) if manifest.exists() else None
    if ledger is not None and ledger.get("signature") != signature:
        raise RuntimeError(
            "Stage inputs or resources changed; refusing to duplicate an existing recovery campaign"
        )
    legacy = None
    if ledger is None or not ledger.get("instrumented", True):
        legacy_builder = dict(
            ledger.get("legacy_builder", original_builder)
            if ledger
            else original_builder
        )
        legacy = make_submission(
            machine, resources, submission_class=StageSubmission, **legacy_builder
        )
        if not isinstance(legacy, StageSubmission):
            raise ValueError(
                "job_recovery requires batch Slurm execution without --gpus"
            )
        legacy.generate_jobs()
        exists = legacy.machine.context.check_file_exists(
            f"{legacy.submission_hash}.json"
        )
        if exists:
            legacy.try_recover_from_json()
            expected = {}
            for name in paths:
                for filename in input_files:
                    local = root / name / filename
                    files = sorted(local.rglob("*")) if local.is_dir() else [local]
                    for item in files:
                        if item.is_file():
                            relative = str(item.relative_to(root))
                            expected[relative] = hashlib.sha256(
                                item.read_bytes()
                            ).hexdigest()
            script = (
                "import hashlib,json; from pathlib import Path; "
                f"print(json.dumps({{p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in {list(expected)!r}}}))"
            )
            code, _, stdout, _ = legacy.machine.context.block_call(
                "python3 -c " + quote(script)
            )
            from dpgen.dispatcher.fp_recovery import _text

            if code or json.loads(_text(stdout)) != expected:
                raise RuntimeError(
                    "Existing stage inputs differ; refusing to overwrite or resubmit"
                )
            if any(len(job.job_task_list) != 1 for job in legacy.belonging_jobs):
                raise RuntimeError(
                    "Existing grouped submission cannot be adopted safely; recovery currently requires one task per existing job"
                )
        elif ledger is not None:
            raise RuntimeError(
                "Existing recovery submission record is missing; refusing to create replacement jobs"
            )
        else:
            legacy = None
    if ledger is None:
        ledger = {
            "version": 1,
            "signature": signature,
            "instrumented": legacy is None,
            "stage": stage,
            "tasks": {
                name: {"status": "pending", "input_digest": fingerprints[name]}
                for name in paths
            },
        }
        if legacy is not None:
            ledger["legacy_builder"] = original_builder
        worker.atomic_json(manifest, ledger)
    for name in paths:
        task = root / name
        shutil.copyfile(worker.__file__, task / "fp_recovery_worker.py")
        policy = {"command": command, "input_files": input_files, "bad_gpu_uuids": []}
        worker.atomic_json(task / "fp-recovery.json", policy)
    arguments = dict(builder)
    arguments.update(
        commands=["python fp_recovery_worker.py"],
        group_size=1,
        forward_files=list(builder["forward_files"])
        + ["fp_recovery_worker.py", "fp-recovery.json"],
        backward_files=list(builder["backward_files"])
        + ["fp-attempt.json", "fp-runtime.log"],
        submission_class=StageSubmission,
    )
    submission = legacy or make_submission(
        machine, dict(resources, group_size=1), **arguments
    )
    if not isinstance(submission, StageSubmission):
        raise ValueError("job_recovery requires batch Slurm execution without --gpus")
    if stage == "train":
        artifacts = [
            name
            for name in builder["backward_files"]
            if name.startswith("frozen_model")
        ]
        if not artifacts:
            raise ValueError("Training recovery needs an exported frozen model output")
    else:
        artifacts = [
            name
            for name in builder["backward_files"]
            if name.startswith("model_devi") and name.endswith(".out")
        ]
        if not artifacts:
            raise ValueError("Exploration recovery needs committee deviation output")
    plasma = "plasma.json" in input_files
    submission.configure_recovery(
        root,
        config,
        ledger,
        validator=lambda task: validate_result(
            task,
            fingerprints[Path(task).name],
            artifacts,
            plasma,
            require_receipt=ledger["instrumented"],
        ),
        instrumented=ledger["instrumented"],
    )
    submission.recovery_log_file = builder["outlog"]
    if not ledger["instrumented"]:
        dlog.info(
            "%s: adopting existing submission %s in %s; preserving jobs and checkpoints. Existing jobs have no GPU wrapper telemetry.",
            stage,
            submission.submission_hash,
            submission.machine.context.remote_root,
        )
    # Preserve the staged inputs and all remote checkpoints on retries.
    # Uploading initial checkpoint directories would roll back MD.
    submission.preserve_remote_checkpoints = True
    dlog.info("%s: hardware-aware recovery enabled; one task per Slurm job", stage)
    return submission
