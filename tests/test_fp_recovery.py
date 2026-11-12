"""Evidence-based FP retry, device allocation boundaries, and partial recovery."""

import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dpdispatcher.submission import JobStatus

from dpgen.dispatcher import fp_recovery as recovery
from dpgen.dispatcher import fp_recovery_worker as worker


class MemoryContext:
    remote_root = "/persistent/submission"

    def __init__(self):
        self.files = {}
        self.downloads = []
        self.commands = []
        self.accounting = "1|FAILED|1:0|ghpc011|gpu|None\n"

    def read_file(self, fname):
        if str(fname) not in self.files:
            raise FileNotFoundError(fname)
        return self.files[str(fname)]

    def write_file(self, fname, write_str):
        self.files[str(fname)] = write_str

    def download(self, submission, **kwargs):
        self.downloads.append(
            ([task.task_work_path for task in submission.belonging_tasks], kwargs)
        )

    def block_call(self, command):
        self.commands.append(command)
        return (
            0,
            None,
            io.StringIO(self.accounting if command.startswith("sacct") else ""),
            io.StringIO(""),
        )


def task_job(name, job_id="1", state=JobStatus.terminated):
    task = SimpleNamespace(task_work_path=name, task_hash="a" * 40, task_state=state)
    job = SimpleNamespace(
        job_task_list=[task],
        job_hash="b" * 40,
        job_id=job_id,
        job_state=state,
        fail_count=0,
        failure_reason=None,
    )
    job.get_last_error_message = Mock(return_value="")
    job._ensure_forward_files_on_retry = Mock()
    job.get_job_state = Mock()

    def submit():
        job.job_id = str(int(job.job_id) + 1)
        job.job_state = JobStatus.waiting

    job.submit_job = Mock(side_effect=submit)
    return task, job


class TestFPRecovery(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.original = os.getcwd()
        os.chdir(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(os.chdir, self.original)
        self.root = Path("iter.000001/02.fp")
        self.root.mkdir(parents=True)

    def submission(self, jobs, instrumented=False):
        submission = recovery.RecoverySubmission.__new__(recovery.RecoverySubmission)
        submission.belonging_jobs = jobs
        submission.belonging_tasks = [job.job_task_list[0] for job in jobs]
        submission.submission_hash = "c" * 40
        context = MemoryContext()
        submission.machine = SimpleNamespace(context=context)
        ledger = {
            "signature": "test",
            "tasks": {
                task.task_work_path: {"status": "pending"}
                for task in submission.belonging_tasks
            },
        }
        for name in ledger["tasks"]:
            (self.root / name).mkdir(exist_ok=True)
            worker.atomic_json(
                self.root / name / "fp-recovery.json",
                {"command": "vasp", "bad_gpu_uuids": []},
            )
        config = dict(
            recovery.DEFAULTS,
            enabled=True,
            hardware_retry_delay=0,
            nonhardware_retry_delay=0,
        )
        submission.configure_recovery(
            self.root,
            config,
            ledger,
            validator=lambda path: (True, "valid"),
            instrumented=instrumented,
        )
        submission.submission_to_json = Mock()
        return submission

    def test_hardware_retries_have_no_attempt_limit(self):
        task, job = task_job("task.019.000000")
        submission = self.submission([job], instrumented=True)
        submission._diagnose = Mock(
            return_value={
                "category": "hardware",
                "reason": "uncorrectable ECC",
                "scheduler": {"node": "ghpc011", "partition": "gpu"},
                "attempt": {
                    "node": "ghpc011",
                    "partition": "gpu",
                    "faulty_gpu_uuids": ["GPU-bad"],
                },
            }
        )
        for _ in range(12):  # Deliberately exceed the backend's generic retry limit.
            job.job_state = JobStatus.terminated
            submission.handle_unexpected_submission_state(continue_on_failure=True)
        self.assertEqual(job.submit_job.call_count, 12)
        self.assertEqual(
            submission.recovery_ledger["tasks"][task.task_work_path][
                "hardware_failures"
            ],
            12,
        )
        self.assertEqual(job.fail_count, 12)
        self.assertIn("GPU-bad", submission.health["devices"])
        remote = submission.machine.context.files[
            task.task_work_path + "/fp-recovery.json"
        ]
        self.assertEqual(json.loads(remote)["bad_gpu_uuids"], ["GPU-bad"])
        history = (self.root / "fp-recovery-history.jsonl").read_text().splitlines()
        self.assertEqual(len(history), 12)

    def test_legacy_failed_alias_does_not_skip_terminated_jobs(self):
        task, job = task_job("task.019.000000")
        submission = self.submission([job])
        submission._diagnose = Mock(
            return_value={
                "category": "hardware",
                "reason": "ECC error",
                "scheduler": {},
                "attempt": {},
            }
        )
        with patch.object(recovery, "FAILED", JobStatus.terminated):
            submission.handle_unexpected_submission_state(continue_on_failure=True)
        self.assertEqual(job.submit_job.call_count, 1)
        self.assertEqual(
            submission.recovery_ledger["tasks"][task.task_work_path][
                "hardware_failures"
            ],
            1,
        )

    def test_progress_heartbeat_records_waiting_and_retry_counts(self):
        task, job = task_job("task.019.000000", state=JobStatus.running)
        other, waiting = task_job("task.020.000000", "2", JobStatus.waiting)
        submission = self.submission([job, waiting])
        submission.recovery_ledger["tasks"][task.task_work_path]["status"] = "running"
        submission.recovery_ledger["tasks"][other.task_work_path]["status"] = (
            "retry_wait"
        )
        with patch.object(recovery.dlog, "info") as log:
            submission._log_progress(30)
        state = json.loads((self.root / ".fp-recovery.json").read_text())
        self.assertEqual(state["counts"], {"running": 1, "retry_wait": 1})
        self.assertEqual(state["activity"]["phase"], "waiting for next scheduler check")
        self.assertIn("next check", log.call_args.args[0])

    def test_nonhardware_bounded_and_running_sibling_not_resubmitted(self):
        task, failing = task_job("task.019.000000")
        _, running = task_job("task.019.000001", "99", JobStatus.running)
        submission = self.submission([failing, running])
        submission._diagnose = Mock(
            return_value={
                "category": "resource",
                "reason": "out of memory",
                "scheduler": {},
                "attempt": {},
            }
        )
        for _ in range(3):
            failing.job_state = JobStatus.terminated
            submission.handle_unexpected_submission_state(continue_on_failure=True)
        self.assertEqual(failing.submit_job.call_count, 2)
        self.assertEqual(failing.job_state, JobStatus.failed)
        self.assertEqual(task.task_state, JobStatus.failed)
        running.submit_job.assert_not_called()
        self.assertEqual(
            submission.recovery_ledger["tasks"][task.task_work_path]["reason"],
            "out of memory",
        )

    def test_successful_results_collected_once_and_preserved(self):
        task, job = task_job("task.019.000000", state=JobStatus.finished)
        _, running = task_job("task.019.000001", "99", JobStatus.running)
        submission = self.submission([job, running])
        submission.update_submission_state()
        submission.update_submission_state()
        self.assertEqual(len(submission.machine.context.downloads), 1)
        self.assertEqual(
            submission.machine.context.downloads[0][0], [task.task_work_path]
        )
        submission.handle_unexpected_submission_state(continue_on_failure=True)
        job.submit_job.assert_not_called()
        running.submit_job.assert_not_called()
        state = json.loads((self.root / ".fp-recovery.json").read_text())
        self.assertEqual(state["tasks"][task.task_work_path]["status"], "successful")

    def test_scheduler_node_failure_and_stale_gpu_evidence(self):
        task, job = task_job("task.019.000000", "123")
        submission = self.submission([job], instrumented=True)
        context = submission.machine.context
        context.accounting = "123|NODE_FAIL|0:0|ghpc011|gpu|Node failure\n"
        result = submission._diagnose(job, task.task_work_path)
        self.assertEqual(result["category"], "hardware")
        context.accounting = "123|FAILED|1:0|ghpc011|gpu|None\n"
        context.files[task.task_work_path + "/fp-attempt.json"] = json.dumps(
            {"job_id": "122", "category": "hardware", "reason": "old ECC error"}
        )
        context.files[task.task_work_path + "/fp-runtime.log"] = (
            "uncorrectable ECC error"
        )
        result = submission._diagnose(job, task.task_work_path)
        self.assertEqual(result["category"], "unknown")
        self.assertEqual(result["attempt"], {})

    def test_unknown_scheduler_state_never_duplicates_job(self):
        _, job = task_job("task.019.000000", state=JobStatus.unknown)
        submission = self.submission([job])
        with self.assertRaisesRegex(RuntimeError, "rather than submitting a duplicate"):
            submission.handle_unexpected_submission_state()
        job.submit_job.assert_not_called()

    def test_gpu_selection_uses_only_allocated_uuids(self):
        inventory = [
            {"index": str(i), "uuid": f"GPU-{i}", "pci_bus_id": str(i)}
            for i in range(4)
        ]
        environment = {"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "2"}
        # Numeric Slurm and NVML indices alone are not reliable identities.
        self.assertEqual(worker.allocated_gpus(inventory, environment), [])
        allocated = worker.allocated_gpus(inventory, environment, ["GPU-2"])
        self.assertEqual(worker.choose_devices(allocated, ["GPU-2"]), [])
        allocated = worker.allocated_gpus(inventory, environment, ["GPU-1", "GPU-2"])
        self.assertEqual(worker.choose_devices(allocated, ["GPU-2"]), ["GPU-1"])
        self.assertEqual(
            worker.allocated_gpus(inventory, {"CUDA_VISIBLE_DEVICES": "GPU-3"})[0][
                "uuid"
            ],
            "GPU-3",
        )

    def test_worker_refuses_bad_single_gpu_without_launching_vasp(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            os.chdir(directory)
            try:
                worker.atomic_json(
                    "fp-recovery.json",
                    {"command": "vasp", "bad_gpu_uuids": ["GPU-bad"]},
                )
                popen = Mock()
                query = Mock(
                    return_value=SimpleNamespace(
                        returncode=0, stdout="0, GPU-bad, 0000:01:00.0\n", stderr=""
                    )
                )
                with patch.dict(
                    os.environ,
                    {
                        "CUDA_VISIBLE_DEVICES": "0",
                        "SLURM_JOB_GPUS": "0",
                        "SLURM_JOB_ID": "1",
                    },
                    clear=True,
                ):
                    with patch.object(
                        worker, "cuda_visible_uuids", return_value=(["GPU-bad"], "")
                    ):
                        self.assertEqual(worker.run_attempt(run=query, popen=popen), 75)
                popen.assert_not_called()
                report = json.loads(Path("fp-attempt.json").read_text())
                self.assertEqual(report["category"], "hardware")
                self.assertEqual(report["allocated_gpus"][0]["uuid"], "GPU-bad")
            finally:
                os.chdir(previous)

    def test_worker_masks_bad_card_but_retains_healthy_allocated_card(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            os.chdir(directory)
            try:
                worker.atomic_json(
                    "fp-recovery.json",
                    {"command": "vasp", "bad_gpu_uuids": ["GPU-bad"]},
                )
                query = Mock(
                    return_value=SimpleNamespace(
                        returncode=0,
                        stdout="0, GPU-bad, 0\n1, GPU-good, 1\n2, GPU-other-job, 2\n",
                        stderr="",
                    )
                )
                process = SimpleNamespace(
                    stdout=iter(["CUDA error: uncorrectable ECC error\n"]),
                    wait=lambda: 1,
                )
                popen = Mock(return_value=process)
                with patch.dict(
                    os.environ,
                    {
                        "CUDA_VISIBLE_DEVICES": "0,1",
                        "SLURM_JOB_GPUS": "0,1",
                        "SLURM_JOB_ID": "1",
                    },
                    clear=True,
                ):
                    with patch.object(
                        worker,
                        "cuda_visible_uuids",
                        return_value=(["GPU-bad", "GPU-good"], ""),
                    ):
                        self.assertEqual(worker.run_attempt(run=query, popen=popen), 1)
                self.assertEqual(
                    popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-good"
                )
                report = json.loads(Path("fp-attempt.json").read_text())
                self.assertEqual(report["faulty_gpu_uuids"], ["GPU-good"])
                self.assertNotIn("GPU-other-job", report["selected_gpu_uuids"])
            finally:
                os.chdir(previous)

    def test_only_explicit_hardware_errors_are_unlimited(self):
        for message in (
            "uncorrectable ECC error",
            "GPU has fallen off the bus",
            "cudaErrorECCUncorrectable",
        ):
            self.assertEqual(worker.classify(message)[0], "hardware")
        for message in (
            "CUDA out of memory",
            "CUDA illegal memory access",
            "MPI_ABORT",
            "BRMIX convergence error",
        ):
            self.assertNotEqual(worker.classify(message)[0], "hardware")
        self.assertEqual(worker.classify("", "NODE_FAIL+")[0], "hardware")

    def test_real_vasp_output_validation_and_changed_geometry(self):
        import dpdata

        fixture = (
            Path(__file__).resolve().parent
            / "generator/out_data_post_fp_vasp/02.fp/task.000.000000/OUTCAR"
        )
        task = self.root / "task.019.000000"
        task.mkdir()
        (task / "OUTCAR").write_bytes(fixture.read_bytes())
        result = dpdata.LabeledSystem(str(task / "OUTCAR"), fmt="vasp/outcar")
        result.to_vasp_poscar(str(task / "POSCAR"))
        self.assertTrue(recovery.validate_vasp(task)[0])
        result.data["coords"][0, 0] += result.data["cells"][0, 0]
        result.to_vasp_poscar(str(task / "POSCAR"))
        self.assertTrue(recovery.validate_vasp(task)[0])
        result.data["coords"][0, 0, 0] += 0.1
        result.to_vasp_poscar(str(task / "POSCAR"))
        self.assertFalse(recovery.validate_vasp(task)[0])

    def test_policy_validation(self):
        for key, value in (
            ("enabled", 1),
            ("max_nonhardware_retries", -1),
            ("hardware_retry_delay", -1),
            ("check_interval", 0),
        ):
            with self.assertRaises(ValueError):
                recovery.options({"fp_recovery": {key: value}})
        self.assertFalse(recovery.options({})["enabled"])

    def test_invalid_labels_are_not_treated_as_scheduler_success(self):
        task, job = task_job("task.019.000000", state=JobStatus.finished)
        submission = self.submission([job])
        submission.recovery_validator = lambda path: (False, "not converged")
        submission.update_submission_state()
        self.assertEqual(job.job_state, JobStatus.terminated)
        self.assertEqual(
            submission.recovery_ledger["tasks"][task.task_work_path][
                "validation_error"
            ],
            "not converged",
        )

    def test_poll_loop_preserves_sibling_and_finishes_after_hardware_retry(self):
        first_task, first = task_job("task.019.000000")
        _, sibling = task_job("task.019.000001", "90", JobStatus.running)
        submission = self.submission([first, sibling])
        submission.try_recover_from_json = Mock()
        submission.upload_jobs = Mock()
        submission.serialize = Mock(return_value={"complete": True})
        submission._diagnose = Mock(
            return_value={
                "category": "hardware",
                "reason": "NODE_FAIL",
                "scheduler": {},
                "attempt": {},
            }
        )
        calls = {"sibling": 0}

        def update_first():
            if first.job_id == "2":
                first.job_state = JobStatus.finished
                first_task.task_state = JobStatus.finished

        def update_sibling():
            calls["sibling"] += 1
            if calls["sibling"] > 1:
                sibling.job_state = JobStatus.finished
                sibling.job_task_list[0].task_state = JobStatus.finished

        first.get_job_state.side_effect = update_first
        sibling.get_job_state.side_effect = update_sibling
        waits = []

        def fake_wait(seconds):
            waits.append(seconds)
            if len(waits) > 5:
                raise AssertionError(
                    "Recovery stalled instead of retrying the terminated job"
                )

        with (
            patch("dpgen.dispatcher.fp_recovery.time.sleep", side_effect=fake_wait),
            patch.object(recovery, "FAILED", JobStatus.terminated),
        ):
            result = submission.run_submission(clean=False, check_interval=1)
        self.assertEqual(result, {"complete": True})
        submission.upload_jobs.assert_not_called()
        self.assertEqual(first.submit_job.call_count, 1)
        sibling.submit_job.assert_not_called()
        self.assertTrue(
            all(
                entry["status"] == "successful"
                for entry in submission.recovery_ledger["tasks"].values()
            )
        )
        self.assertEqual(
            len(submission.machine.context.downloads), 3
        )  # failure diagnostics + two results

    def test_submission_marker_recovers_job_id_before_any_retry(self):
        task, job = task_job("task.019.000000", "1", JobStatus.unsubmitted)
        submission = self.submission([job])
        submission.recovery_ledger["tasks"][task.task_work_path][
            "submission_in_progress"
        ] = True
        submission.machine.context.files[job.job_hash + "_job_id"] = "2"
        submission.update_submission_state()
        self.assertEqual(job.job_id, "2")
        self.assertEqual(job.job_state, JobStatus.waiting)
        submission.handle_unexpected_submission_state()
        job.submit_job.assert_not_called()
        self.assertFalse(
            submission.recovery_ledger["tasks"][task.task_work_path][
                "submission_in_progress"
            ]
        )

    def test_remote_record_ahead_of_local_manifest_recovers_intent(self):
        task, job = task_job("task.019.000000", "2", JobStatus.waiting)
        submission = self.submission([job])
        entry = submission.recovery_ledger["tasks"][task.task_work_path]
        entry.update(submission_in_progress=True, job_id="1")
        submission.machine.context.files[job.job_hash + "_job_id"] = "2"
        submission.update_submission_state()
        self.assertEqual(job.job_id, "2")
        self.assertFalse(entry["submission_in_progress"])
        job.submit_job.assert_not_called()

    def test_ambiguous_submission_stops_instead_of_duplicating(self):
        task, job = task_job("task.019.000000", "1", JobStatus.unsubmitted)
        submission = self.submission([job])
        submission.recovery_ledger["tasks"][task.task_work_path].update(
            submission_in_progress=True, job_id="1"
        )
        submission.machine.context.files[job.job_hash + "_job_id"] = "1"
        with self.assertRaisesRegex(RuntimeError, "uncertain"):
            submission.update_submission_state()
        job.submit_job.assert_not_called()

    def real_machine(self):
        return {
            "batch_type": "Slurm",
            "context_type": "LocalContext",
            "local_root": "./",
            "remote_root": str(Path("remote").resolve()),
        }

    def real_resources(self):
        return {
            "number_node": 1,
            "cpu_per_node": 1,
            "gpu_per_node": 1,
            "group_size": 1,
            "queue_name": "gpu",
        }

    def make_inputs(self, name):
        task = self.root / name
        task.mkdir()
        for filename in ("POSCAR", "INCAR", "POTCAR", "KPOINTS"):
            (task / filename).write_text(filename + "\n")

    def test_adopts_existing_unwrapped_submission_and_running_job(self):
        from dpgen.dispatcher.Dispatcher import make_submission

        name = "task.019.000000"
        self.make_inputs(name)
        files = ["POSCAR", "INCAR", "POTCAR", "KPOINTS"]
        outputs = ["fp.log", "OUTCAR", "vasprun.xml"]
        machine, resources = self.real_machine(), self.real_resources()
        original = make_submission(
            machine,
            resources,
            commands=["vasp"],
            work_path=str(self.root),
            run_tasks=[name],
            group_size=1,
            forward_common_files=[],
            forward_files=files,
            backward_files=outputs,
            outlog="fp.log",
            errlog="fp.log",
        )
        original.generate_jobs()
        original.upload_jobs()
        original.belonging_jobs[0].job_id = "4711"
        original.belonging_jobs[0].job_state = JobStatus.running
        original.belonging_jobs[0].job_task_list[0].task_state = JobStatus.running
        original.submission_to_json()
        recovered = []

        def inspect_recovery(submission, **kwargs):
            submission.try_recover_from_json()
            recovered.append(submission)
            self.assertEqual(kwargs["clean"], False)
            self.assertEqual(kwargs["continue_on_failure"], True)

        data = {"fp_style": "vasp", "fp_recovery": {"enabled": True}}
        mdata = {
            "fp_command": "vasp",
            "fp_machine": machine,
            "fp_resources": resources,
            "fp_group_size": 1,
        }
        with (
            patch.object(
                recovery.RecoverySubmission,
                "run_submission",
                autospec=True,
                side_effect=inspect_recovery,
            ),
            patch.object(recovery, "validate_vasp", return_value=(False, "no result")),
        ):
            recovery.run_recovery(self.root, data, mdata, files, outputs)
        self.assertEqual(recovered[0].submission_hash, original.submission_hash)
        self.assertEqual(
            recovered[0].machine.context.remote_root,
            original.machine.context.remote_root,
        )
        self.assertEqual(recovered[0].belonging_jobs[0].job_id, "4711")
        self.assertEqual(recovered[0].belonging_jobs[0].job_state, JobStatus.running)
        self.assertFalse(recovered[0].instrumented)

    def test_new_campaign_skips_completed_task_and_reuses_wrapped_identity(self):
        name, finished = "task.019.000000", "task.019.000001"
        self.make_inputs(name)
        self.make_inputs(finished)
        files = ["POSCAR", "INCAR", "POTCAR", "KPOINTS"]
        outputs = ["fp.log", "OUTCAR", "vasprun.xml"]
        mdata = {
            "fp_command": "vasp",
            "fp_machine": self.real_machine(),
            "fp_resources": self.real_resources(),
            "fp_group_size": 1,
        }
        data = {"fp_style": "vasp", "fp_recovery": {"enabled": True}}
        submissions = []

        def simulate(submission, **kwargs):
            submission.generate_jobs()
            submission.try_recover_from_json()
            if not submissions:
                submission.upload_jobs()
                submission.belonging_jobs[0].job_id = "4811"
                submission.belonging_jobs[0].job_state = JobStatus.running
                submission.submission_to_json()
            submissions.append(submission)
            submission._save()

        with (
            patch.object(
                recovery.RecoverySubmission,
                "run_submission",
                autospec=True,
                side_effect=simulate,
            ),
            patch.object(
                recovery,
                "validate_vasp",
                side_effect=lambda path: (Path(path).name == finished, "checked"),
            ),
        ):
            recovery.run_recovery(self.root, data, mdata, files, outputs)
            recovery.run_recovery(self.root, data, mdata, files, outputs)
        self.assertEqual(submissions[0].submission_hash, submissions[1].submission_hash)
        self.assertEqual(
            submissions[0].machine.context.remote_root,
            submissions[1].machine.context.remote_root,
        )
        self.assertEqual(submissions[1].belonging_jobs[0].job_id, "4811")
        self.assertEqual(
            [task.task_work_path for task in submissions[1].belonging_tasks], [name]
        )
        self.assertTrue(submissions[1].instrumented)
        # Changed scientific inputs must never silently inherit success or cause
        # submission into another directory while the original job is running.
        (self.root / name / "INCAR").write_text("different settings")
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite or duplicate"):
            recovery.run_recovery(self.root, data, mdata, files, outputs)

    def test_legacy_bind_and_deserialize_signatures_preserve_existing_job(self):
        from dpdispatcher import Submission

        from dpgen.dispatcher.Dispatcher import make_submission

        name = "task.019.000000"
        self.make_inputs(name)
        original = make_submission(
            self.real_machine(),
            self.real_resources(),
            commands=["vasp"],
            work_path=str(self.root),
            run_tasks=[name],
            group_size=1,
            forward_common_files=[],
            forward_files=["POSCAR", "INCAR", "POTCAR", "KPOINTS"],
            backward_files=["OUTCAR"],
            outlog="fp.log",
            errlog="fp.log",
        )
        original.generate_jobs()
        original.upload_jobs()
        original.belonging_jobs[0].job_id = "8811"
        original.belonging_jobs[0].job_state = JobStatus.running
        original.belonging_jobs[0].job_task_list[0].task_state = JobStatus.running
        original.submission_to_json()
        directory = original.machine.context.remote_root
        record = original.serialize()

        def legacy_bind(instance, machine):
            raise AssertionError("Legacy binding must be handled by the adapter")

        def legacy_deserialize(cls, submission_dict):
            raise AssertionError("Do not use the incompatible vendor deserializer")

        with (
            patch.object(Submission, "bind_machine", legacy_bind),
            patch.object(Submission, "deserialize", classmethod(legacy_deserialize)),
        ):
            restored = recovery.RecoverySubmission.deserialize(
                record, machine=original.machine, bind_context=False
            )
            self.assertEqual(original.machine.context.remote_root, directory)
            restored.existing_remote_root = directory
            restored.bind_machine(original.machine)
            restored.try_recover_from_json()
            self.assertEqual(restored.submission_hash, original.submission_hash)
            self.assertEqual(restored.belonging_jobs[0].job_id, "8811")
            self.assertEqual(restored.belonging_jobs[0].job_state, JobStatus.running)
            self.assertEqual(restored.machine.context.remote_root, directory)
            # Also exercise the constructor path from the user's traceback.
            fresh = make_submission(
                self.real_machine(),
                self.real_resources(),
                commands=["vasp"],
                work_path=str(self.root),
                run_tasks=[name],
                group_size=1,
                forward_common_files=[],
                forward_files=["POSCAR"],
                backward_files=["OUTCAR"],
                outlog="fp.log",
                errlog="fp.log",
                submission_class=recovery.RecoverySubmission,
            )
            self.assertIsInstance(fresh, recovery.RecoverySubmission)

    def test_cli_existing_dir_is_scoped_to_one_fp_stage(self):
        from dpgen.main import main_parser

        directory = str(Path("old-stage").resolve())
        for command in ("run", "finetune"):
            args = main_parser().parse_args(
                [command, "param.json", "machine.json", "--fp-existing-dir", directory]
            )
            with recovery.recovery_execution(args):
                self.assertEqual(recovery.requested_directory(self.root), directory)
                self.assertEqual(recovery.requested_directory(self.root), directory)
                self.assertIsNone(recovery.requested_directory("iter.000002/02.fp"))
            self.assertIsNone(recovery.requested_directory(self.root))

    def test_explicit_directory_pins_original_record_and_never_creates_replacement(
        self,
    ):
        from dpgen.dispatcher.Dispatcher import make_submission

        name = "task.019.000000"
        self.make_inputs(name)
        files = ["POSCAR", "INCAR", "POTCAR", "KPOINTS"]
        outputs = ["fp.log", "OUTCAR", "vasprun.xml"]
        machine, resources = self.real_machine(), self.real_resources()
        original = make_submission(
            machine,
            resources,
            commands=["vasp"],
            work_path=str(self.root),
            run_tasks=[name],
            group_size=1,
            forward_common_files=[],
            forward_files=files,
            backward_files=outputs,
            outlog="fp.log",
            errlog="fp.log",
        )
        original.generate_jobs()
        original.upload_jobs()
        original.belonging_jobs[0].job_id = "7711"
        original.belonging_jobs[0].job_state = JobStatus.running
        original.belonging_jobs[0].job_task_list[0].task_state = JobStatus.running
        original.submission_to_json()
        calculated = Path(original.machine.context.remote_root)
        selected_directory = Path("existing-old-fp").resolve()
        shutil.move(str(calculated), selected_directory)
        resumed = []

        def inspect_resume(submission, **kwargs):
            submission.try_recover_from_json()
            submission.upload_jobs()
            submission._save()
            resumed.append(submission)

        data = {"fp_style": "vasp", "fp_recovery": {"enabled": True}}
        mdata = {
            "fp_command": "vasp",
            "fp_machine": machine,
            "fp_resources": resources,
            "fp_group_size": 1,
        }
        args = SimpleNamespace(fp_existing_dir=str(selected_directory))
        with (
            patch.object(
                recovery.RecoverySubmission,
                "run_submission",
                autospec=True,
                side_effect=inspect_resume,
            ),
            patch.object(
                recovery, "validate_vasp", return_value=(False, "not local yet")
            ),
        ):
            with recovery.recovery_execution(args):
                recovery.run_recovery(self.root, data, mdata, files, outputs)
            # The pinned directory persists in the local manifest for later resumes.
            recovery.run_recovery(self.root, data, mdata, files, outputs)
        self.assertEqual(len(resumed), 2)
        self.assertTrue(
            all(
                item.machine.context.remote_root == str(selected_directory)
                for item in resumed
            )
        )
        self.assertTrue(
            all(item.belonging_jobs[0].job_id == "7711" for item in resumed)
        )
        self.assertTrue(
            all(item.submission_hash == original.submission_hash for item in resumed)
        )
        self.assertFalse(calculated.exists())
        manifest = json.loads((self.root / ".fp-recovery.json").read_text())
        self.assertEqual(manifest["existing_directory"], str(selected_directory))
        # Explicitly selecting a nonexistent directory must not silently fall
        # back to the calculated hash directory or create any replacement.
        with recovery.recovery_execution(
            SimpleNamespace(fp_existing_dir=str(Path("missing").resolve()))
        ):
            with self.assertRaises((FileNotFoundError, ValueError, OSError)):
                recovery.run_recovery(self.root, data, mdata, files, outputs)
        self.assertFalse(Path("missing").exists())
        self.assertFalse(calculated.exists())

    def test_explicit_resume_keeps_saved_resources_when_current_settings_differ(self):
        from dpgen.dispatcher.Dispatcher import make_submission

        name = "task.019.000000"
        self.make_inputs(name)
        common = dict(
            commands=["vasp"],
            work_path=str(self.root),
            run_tasks=[name],
            group_size=1,
            forward_common_files=[],
            forward_files=["POSCAR", "INCAR", "POTCAR", "KPOINTS"],
            backward_files=["OUTCAR"],
            outlog="fp.log",
            errlog="fp.log",
        )
        original = make_submission(self.real_machine(), self.real_resources(), **common)
        original.generate_jobs()
        original.upload_jobs()
        original.belonging_jobs[0].job_id = "9911"
        original.belonging_jobs[0].job_state = JobStatus.running
        original.submission_to_json()
        directory = original.machine.context.remote_root
        changed = dict(
            self.real_resources(), cpu_per_node=4, queue_name="another-partition"
        )
        probe = make_submission(
            self.real_machine(),
            changed,
            submission_class=recovery.RecoverySubmission,
            **common,
        )
        probe.generate_jobs()
        replacement = probe.machine.context.remote_root
        with patch.object(recovery.dlog, "warning") as warning:
            selected, wrapped, location = recovery._load_existing(
                probe, directory, self.root.resolve(), [name], "vasp"
            )
        self.assertFalse(wrapped)
        self.assertEqual(selected.submission_hash, original.submission_hash)
        self.assertEqual(selected.belonging_jobs[0].job_id, "9911")
        self.assertEqual(selected.resources.cpu_per_node, 1)
        self.assertEqual(selected.resources.queue_name, "gpu")
        self.assertEqual(selected.belonging_jobs[0].resources.queue_name, "gpu")
        self.assertEqual(
            set(selected.resource_difference_fields), {"cpu_per_node", "queue_name"}
        )
        self.assertEqual(location, directory)
        self.assertFalse(Path(replacement).exists())
        warning.assert_called_once()
        self.assertEqual(changed["queue_name"], "another-partition")

    def test_exclusive_campaign_lock(self):
        import fcntl

        with (self.root / ".fp-recovery.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Another controller"):
                recovery.run_recovery(self.root, {}, {}, [], [])

    def test_cuda_context_unknown_is_initialization_not_confirmed_hardware(self):
        message = "Accelerator Fatal Error: call to cuCtxCreate returned error 999 (CUDA_ERROR_UNKNOWN): Unknown"
        self.assertEqual(worker.classify(message)[0], "gpu_initialization")
        self.assertEqual(worker.classify("CUDA_ERROR_UNKNOWN: 999")[0], "unknown")
        self.assertNotIn(
            worker.classify("cuCtxCreate returned CUDA_ERROR_OUT_OF_MEMORY")[0],
            worker.UNLIMITED_RETRY_CATEGORIES,
        )
        self.assertNotIn(
            worker.classify("CUDA illegal memory access")[0],
            worker.UNLIMITED_RETRY_CATEGORIES,
        )

    def test_gpu_initialization_retries_are_unlimited_without_blacklisting(self):
        task, job = task_job("task.019.000000")
        submission = self.submission([job], instrumented=True)
        submission._diagnose = Mock(
            return_value={
                "category": "gpu_initialization",
                "reason": "CUDA context creation failed with error 999",
                "scheduler": {"node": "ghpc011"},
                "attempt": {"faulty_gpu_uuids": ["GPU-unconfirmed"]},
            }
        )
        for _ in range(12):
            job.job_state = JobStatus.terminated
            submission.handle_unexpected_submission_state(continue_on_failure=True)
        self.assertEqual(job.submit_job.call_count, 12)
        entry = submission.recovery_ledger["tasks"][task.task_work_path]
        self.assertEqual(entry["gpu_initialization_failures"], 12)
        self.assertEqual(entry.get("nonhardware_failures", 0), 0)
        self.assertEqual(submission.health["devices"], {})

    def test_exhausted_unknown_context_failure_is_reopened_without_new_job_identity(
        self,
    ):
        task, job = task_job("task.019.000000")
        submission = self.submission([job], instrumented=True)
        ledger = submission.recovery_ledger
        entry = ledger["tasks"][task.task_work_path]
        entry.update(
            status="failed",
            nonhardware_failures=3,
            job_id="1",
            last_failure_job_id="1",
            diagnosis={
                "task": task.task_work_path,
                "job_id": "1",
                "category": "unknown",
                "scheduler": {},
                "attempt": {},
                "evidence": "call to cuCtxCreate returned error 999 (CUDA_ERROR_UNKNOWN): Unknown",
            },
        )
        successful = self.root / "completed-output"
        successful.write_text("preserved")
        submission.configure_recovery(
            self.root, submission.recovery_options, ledger, instrumented=True
        )
        self.assertEqual(entry["status"], "retry_wait")
        self.assertEqual(entry["diagnosis"]["category"], "gpu_initialization")
        self.assertEqual(entry["job_id"], "1")
        self.assertEqual(successful.read_text(), "preserved")
        submission.handle_unexpected_submission_state(continue_on_failure=True)
        job.submit_job.assert_called_once()
        self.assertEqual(submission.health["devices"], {})
        # Reopening must happen only once; no repeated counter adjustments.
        submission.configure_recovery(
            self.root, submission.recovery_options, ledger, instrumented=True
        )
        self.assertEqual(entry["nonhardware_failures"], 2)
        self.assertEqual(entry["gpu_initialization_failures"], 1)

    def cuda_runtime(self, bus="0000:65:00.0", status=0):
        def count(pointer):
            pointer._obj.value = 1
            return status

        def pci(buffer, length, ordinal):
            self.assertEqual(ordinal, 0)  # CUDA-local, not SLURM_JOB_GPUS/NVML.
            buffer.value = bus.encode()
            return 0

        return SimpleNamespace(
            cudaGetDeviceCount=Mock(side_effect=count),
            cudaDeviceGetPCIBusId=Mock(side_effect=pci),
        )

    def test_cuda_pci_lookup_maps_local_zero_to_physical_uuid(self):
        runtime = self.cuda_runtime()
        inventory = [
            {
                "index": "0",
                "uuid": "GPU-other",
                "pci_bus_id": "00000000:12:00.0",
                "mig_mode": "Disabled",
            },
            {
                "index": "7",
                "uuid": "GPU-allocated",
                "pci_bus_id": "00000000:65:00.0",
                "mig_mode": "Disabled",
            },
        ]
        with (
            patch.object(worker.ctypes, "CDLL", return_value=runtime),
            patch.object(
                worker.ctypes.util, "find_library", return_value="libcudart.so"
            ),
        ):
            result, error = worker.cuda_visible_uuids(
                {"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "1"}, inventory
            )
        self.assertEqual(result, ["GPU-allocated"])
        self.assertEqual(error, "")
        self.assertFalse(hasattr(runtime, "cudaDeviceGetUuid"))

    def test_failed_cuda_enumeration_does_not_guess_physical_index(self):
        with (
            patch.object(
                worker.ctypes, "CDLL", return_value=self.cuda_runtime(status=999)
            ),
            patch.object(
                worker.ctypes.util, "find_library", return_value="libcudart.so"
            ),
        ):
            result, error = worker.cuda_visible_uuids(
                {"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "1"},
                [
                    {
                        "index": "1",
                        "uuid": "GPU-not-verified",
                        "pci_bus_id": "00000000:65:00.0",
                        "mig_mode": "Disabled",
                    }
                ],
            )
        self.assertIsNone(result)
        self.assertIn("999", error)

    def test_pci_lookup_never_widens_mig_allocation(self):
        for mode in ("Enabled", None):
            with (
                self.subTest(mode=mode),
                patch.object(worker.ctypes, "CDLL", return_value=self.cuda_runtime()),
                patch.object(
                    worker.ctypes.util, "find_library", return_value="libcudart.so"
                ),
            ):
                result, error = worker.cuda_visible_uuids(
                    {"CUDA_VISIBLE_DEVICES": "0"},
                    [
                        {
                            "index": "0",
                            "uuid": "GPU-parent",
                            "pci_bus_id": "00000000:65:00.0",
                            "mig_mode": mode,
                        }
                    ],
                )
            self.assertIsNone(result)
            self.assertIn("MIG", error)
        self.assertEqual(
            worker.cuda_visible_uuids({"CUDA_VISIBLE_DEVICES": "MIG-assigned"}, []),
            (["MIG-assigned"], ""),
        )

    def test_retry_refreshes_old_remote_worker(self):
        task, job = task_job("task.019.000000", state=JobStatus.unsubmitted)
        submission = self.submission([job], instrumented=True)
        submission.machine.context.files[
            task.task_work_path + "/fp_recovery_worker.py"
        ] = "old worker"
        submission.handle_unexpected_submission_state(continue_on_failure=True)
        updated = submission.machine.context.files[
            task.task_work_path + "/fp_recovery_worker.py"
        ]
        self.assertIn("cudaDeviceGetPCIBusId", updated)
        self.assertNotIn("runtime.cudaDeviceGetUuid", updated)
        job.submit_job.assert_called_once()

    def test_scientific_stop_is_discarded_without_retry(self):
        task, job = task_job("task.019.000000")
        submission = self.submission([job], instrumented=True)
        submission._diagnose = Mock(
            return_value={
                "category": "scientific_stop",
                "reason": "SCF stagnation",
                "scheduler": {},
                "attempt": {},
            }
        )
        submission.handle_unexpected_submission_state(continue_on_failure=True)
        entry = submission.recovery_ledger["tasks"][task.task_work_path]
        self.assertEqual(entry["status"], "discarded")
        self.assertEqual(entry["scientific_stops"], 1)
        job.submit_job.assert_not_called()
        submission.handle_unexpected_submission_state(continue_on_failure=True)
        job.submit_job.assert_not_called()
        self.assertEqual(submission.health["devices"], {})

    def test_monitor_stop_marker_survives_interrupted_worker_report(self):
        task, job = task_job("task.019.000000")
        submission = self.submission([job], instrumented=True)
        submission.fp_monitor_options = {"enabled": True}
        submission.machine.context.files[task.task_work_path + "/fp-attempt.json"] = (
            json.dumps({"job_id": "1", "status": "running"})
        )
        submission.machine.context.files[
            task.task_work_path + "/fp-monitor-stop.json"
        ] = json.dumps({"job_id": "1", "reason": "SCF stagnation"})
        self.assertEqual(
            submission._diagnose(job, task.task_work_path)["category"],
            "scientific_stop",
        )
        submission.machine.context.files[
            task.task_work_path + "/fp-monitor-stop.json"
        ] = json.dumps({"job_id": "old", "reason": "SCF stagnation"})
        self.assertEqual(
            submission._diagnose(job, task.task_work_path)["category"], "unknown"
        )

    def test_all_scientific_stops_complete_dispatch_without_retry(self):
        task, job = task_job("task.019.000000")
        submission = self.submission([job], instrumented=True)
        submission.try_recover_from_json = Mock()
        submission.update_submission_state = Mock()
        submission.serialize = Mock(return_value={"preserved": True})
        submission._diagnose = Mock(
            return_value={
                "category": "scientific_stop",
                "reason": "SCF limit",
                "scheduler": {},
                "attempt": {},
            }
        )
        result = submission.run_submission()
        self.assertEqual(result, {"preserved": True})
        self.assertEqual(
            submission.recovery_ledger["tasks"][task.task_work_path]["status"],
            "discarded",
        )
        job.submit_job.assert_not_called()

    def test_enabling_monitor_preserves_existing_instrumented_campaign(self):
        self.make_inputs("task.019.000000")
        mdata = {
            "fp_command": "vasp",
            "fp_machine": self.real_machine(),
            "fp_resources": self.real_resources(),
        }
        data = {"fp_style": "vasp", "fp_recovery": {"enabled": True}}
        submissions = []

        def simulate(submission, **kwargs):
            submission.generate_jobs()
            submission.try_recover_from_json()
            if not submissions:
                submission.upload_jobs()
                submission.belonging_jobs[0].job_id = "9988"
                submission.belonging_jobs[0].job_state = JobStatus.running
                submission.submission_to_json()
            submissions.append(submission)
            submission._save()

        with patch.object(
            recovery.RecoverySubmission,
            "run_submission",
            autospec=True,
            side_effect=simulate,
        ):
            recovery.run_recovery(
                self.root,
                data,
                mdata,
                ["POSCAR", "INCAR", "POTCAR", "KPOINTS"],
                ["OUTCAR", "fp.log"],
            )
            recovery.run_recovery(
                self.root,
                dict(data, fp_monitor={"enabled": True}),
                mdata,
                ["POSCAR", "INCAR", "POTCAR", "KPOINTS"],
                ["OUTCAR", "fp.log"],
            )
        self.assertEqual(submissions[0].submission_hash, submissions[1].submission_hash)
        self.assertEqual(submissions[1].belonging_jobs[0].job_id, "9988")
        self.assertTrue(submissions[1].fp_monitor_options["enabled"])
