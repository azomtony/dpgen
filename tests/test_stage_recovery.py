"""Training/exploration recovery identity, validation, and checkpoint safety."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dpdispatcher.submission import JobStatus

from dpgen.dispatcher import fp_recovery_worker as worker
from dpgen.dispatcher.stage_recovery import make_stage_submission, validate_result
from tests import test_fp_recovery as fp_tests


class TestStageRecovery(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        previous = os.getcwd()
        os.chdir(temporary.name)
        self.addCleanup(temporary.cleanup)
        self.addCleanup(os.chdir, previous)
        self.root = Path("iter.000001/00.train")
        self.task = self.root / "000"
        self.task.mkdir(parents=True)
        (self.task / "input.json").write_text("{}")
        self.machine = {
            "batch_type": "Slurm",
            "context_type": "LocalContext",
            "local_root": "./",
            "remote_root": str(Path("remote").resolve()),
        }
        self.resources = {
            "number_node": 1,
            "cpu_per_node": 1,
            "gpu_per_node": 1,
            "group_size": 10,
            "queue_name": "gpu",
        }
        self.builder = dict(
            commands=["dp train input.json", "dp freeze"],
            work_path=str(self.root),
            run_tasks=["000"],
            group_size=10,
            forward_common_files=[],
            forward_files=["input.json"],
            backward_files=["frozen_model.pt"],
            outlog="train.log",
            errlog="train.log",
        )
        self.data = {"job_recovery": {"enabled": True, "check_interval": 7}}

    def make(self):
        return make_stage_submission(
            self.data, "train", self.machine, self.resources, **self.builder
        )

    def test_existing_scheduler_identity_is_restored(self):
        first = self.make()
        first.generate_jobs()
        first.upload_jobs()
        job = first.belonging_jobs[0]
        job.job_id = "12345"
        job.job_state = JobStatus.running
        first.submission_to_json()
        first._save()
        second = self.make()
        second.generate_jobs()
        second.try_recover_from_json()
        self.assertEqual(second.belonging_jobs[0].job_id, "12345")
        self.assertEqual(first.submission_hash, second.submission_hash)
        self.assertEqual(
            first.machine.context.remote_root, second.machine.context.remote_root
        )
        self.assertEqual(second.resources.group_size, 1)
        self.assertTrue(second.preserve_remote_checkpoints)

    def test_changed_input_refuses_new_campaign(self):
        self.make()
        (self.task / "input.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(RuntimeError, "refusing to duplicate"):
            self.make()

    def test_disable_preserves_original_grouping(self):
        result = make_stage_submission(
            {}, "train", self.machine, self.resources, **self.builder
        )
        self.assertEqual(result.resources.group_size, 10)
        self.assertFalse((self.task / "fp_recovery_worker.py").exists())

    def test_stage_validation_rejects_stale_output(self):
        (self.task / "frozen_model.pt").write_bytes(b"model")
        self.assertFalse(validate_result(self.task, "digest", ["frozen_model.pt"])[0])
        worker.atomic_json(
            self.task / "fp-attempt.json",
            {"input_digest": "old", "status": "exited", "exit_code": 0},
        )
        self.assertFalse(validate_result(self.task, "digest", ["frozen_model.pt"])[0])
        worker.atomic_json(
            self.task / "fp-attempt.json",
            {"input_digest": "digest", "status": "exited", "exit_code": 0},
        )
        self.assertTrue(validate_result(self.task, "digest", ["frozen_model.pt"])[0])
        (self.task / "frozen_model.pt").write_bytes(b"")
        self.assertFalse(validate_result(self.task, "digest", ["frozen_model.pt"])[0])

    def test_plasma_trust_stop_is_successful_completion(self):
        worker.atomic_json(
            self.task / "fp-attempt.json",
            {"input_digest": "digest", "status": "exited", "exit_code": 0},
        )
        (self.task / "model_devi.out").write_text("# header\n0 0 0 0 0.1 0 0\n")
        worker.atomic_json(self.task / "plasma.status.json", {"status": "running"})
        self.assertFalse(
            validate_result(self.task, "digest", ["model_devi.out"], True)[0]
        )
        worker.atomic_json(self.task / "plasma.status.json", {"status": "trust_stop"})
        self.assertTrue(
            validate_result(self.task, "digest", ["model_devi.out"], True)[0]
        )

    def test_directory_inputs_hash_content_but_not_checkpoint(self):
        (self.task / "old").mkdir()
        model = self.task / "old/model.pt"
        model.write_bytes(b"first")
        first = worker.input_digest(self.task, ["input.json", "old"])
        (self.task / "model.ckpt.pt").write_bytes(b"checkpoint")
        self.assertEqual(first, worker.input_digest(self.task, ["input.json", "old"]))
        model.write_bytes(b"different")
        self.assertNotEqual(
            first, worker.input_digest(self.task, ["input.json", "old"])
        )

    def test_configurable_poll_interval(self):
        submission = self.make()
        with patch(
            "dpgen.dispatcher.fp_recovery.RecoverySubmission.run_submission"
        ) as run:
            submission.run_submission()
        self.assertEqual(run.call_args.kwargs["check_interval"], 7)

    def test_fault_retry_does_not_upload_old_checkpoints(self):
        # Exercise the shared scheduler retry handler with checkpoint protection.
        fixture = fp_tests.TestFPRecovery()
        fixture.root = self.root
        _, job = fp_tests.task_job("000")
        submission = fixture.submission([job], instrumented=True)
        submission.preserve_remote_checkpoints = True
        submission.machine.context.files["000/fp-attempt.json"] = json.dumps(
            {
                "job_id": "1",
                "category": "hardware",
                "reason": "uncorrectable ECC",
                "faulty_gpu_uuids": ["GPU-bad"],
            }
        )
        submission.handle_unexpected_submission_state(continue_on_failure=True)
        job._ensure_forward_files_on_retry.assert_not_called()
        job.submit_job.assert_called_once()
        self.assertIn("GPU-bad", submission.health["devices"])

    def test_non_slurm_rejected(self):
        with self.assertRaisesRegex(ValueError, "Slurm"):
            make_stage_submission(
                self.data,
                "train",
                dict(self.machine, batch_type="Shell"),
                self.resources,
                **self.builder,
            )

    def test_plasma_checkpoint_changes_keep_campaign_identity(self):
        (self.task / "plasma.json").write_text("{}")
        (self.task / "checkpoints").mkdir()
        (self.task / "traj").mkdir()
        builder = dict(
            self.builder,
            forward_files=["input.json", "plasma.json", "checkpoints", "traj"],
            backward_files=[
                "model_devi.out",
                "plasma.status.json",
                "checkpoints",
                "traj",
            ],
        )
        first = make_stage_submission(
            self.data, "model_devi", self.machine, self.resources, **builder
        )
        (self.task / "checkpoints/state.json").write_text('{"event": 3}')
        (self.task / "traj/frame.dump").write_text("trajectory")
        second = make_stage_submission(
            self.data, "model_devi", self.machine, self.resources, **builder
        )
        self.assertEqual(first.submission_hash, second.submission_hash)
        policy = json.loads((self.task / "fp-recovery.json").read_text())
        self.assertNotIn("checkpoints", policy["input_files"])

    def test_worker_runs_stage_chain_and_records_matching_receipt(self):
        self.make()
        previous = os.getcwd()
        os.chdir(self.task)
        try:
            policy = json.loads(Path("fp-recovery.json").read_text())
            policy["command"] = "printf model > frozen_model.pt"
            worker.atomic_json("fp-recovery.json", policy)
            with (
                patch.object(worker, "gpu_inventory", return_value=([], "")),
                patch.object(worker, "cuda_visible_uuids", return_value=([], "")),
            ):
                self.assertEqual(worker.run_attempt(), 0)
            fingerprint = worker.input_digest(names=["input.json"])
            self.assertTrue(
                validate_result(Path.cwd(), fingerprint, ["frozen_model.pt"])[0]
            )
        finally:
            os.chdir(previous)

    def test_global_recovery_options_apply_to_fp_with_override(self):
        from dpgen.dispatcher.fp_recovery import options

        self.assertTrue(options(self.data)["enabled"])
        self.assertEqual(options(self.data)["check_interval"], 7)
        self.assertFalse(
            options(dict(self.data, fp_recovery={"enabled": False}))["enabled"]
        )

    def test_common_file_order_does_not_change_campaign_identity(self):
        builder = dict(self.builder, forward_common_files=["b", "a"])
        first = make_stage_submission(
            self.data, "train", self.machine, self.resources, **builder
        )
        builder["forward_common_files"] = ["a", "b"]
        second = make_stage_submission(
            self.data, "train", self.machine, self.resources, **builder
        )
        self.assertEqual(first.submission_hash, second.submission_hash)
