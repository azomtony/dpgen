"""Evidence-based FP retry, device allocation boundaries, and partial recovery."""

import io
import json
import os
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
        with patch("dpgen.dispatcher.fp_recovery.time.sleep"):
            result = submission.run_submission(clean=False, check_interval=1)
        self.assertEqual(result, {"complete": True})
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

    def test_exclusive_campaign_lock(self):
        import fcntl

        with (self.root / ".fp-recovery.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Another controller"):
                recovery.run_recovery(self.root, {}, {}, [], [])
