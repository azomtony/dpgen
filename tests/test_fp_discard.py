"""Manual discard requests must stop retries before cancelling owned jobs."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from dpdispatcher.submission import JobStatus

from dpgen.dispatcher.fp_discard import REQUESTS, read_requests, request_discard
from dpgen.dispatcher.fp_vasp_monitor import stopped_task
from tests import test_fp_recovery as fixtures


class TestFPDiscard(unittest.TestCase):
    def setUp(self):
        previous = os.getcwd()
        temporary = tempfile.TemporaryDirectory()
        os.chdir(temporary.name)
        self.addCleanup(temporary.cleanup)
        self.addCleanup(os.chdir, previous)
        self.root = Path("iter.000001/02.fp")
        self.root.mkdir(parents=True)
        self.fixture = fixtures.TestFPRecovery()
        self.fixture.root = self.root
        self.task, self.job = fixtures.task_job(
            "task.019.000000", state=JobStatus.running
        )
        self.other_task, self.other_job = fixtures.task_job(
            "task.020.000001", job_id="2", state=JobStatus.running
        )
        self.submission = self.fixture.submission(
            [self.job, self.other_job], instrumented=True
        )
        for task, job in ((self.task, self.job), (self.other_task, self.other_job)):
            entry = self.submission.recovery_ledger["tasks"][task.task_work_path]
            entry.update(job_id=job.job_id, status="running", input_digest="d" * 64)
        self.submission._save()
        (self.root / self.task.task_work_path / "POSCAR").write_text(
            "preserved structure"
        )

    def request(self, **kwargs):
        return request_discard(
            self.root, jobs=["1"], reason="Unstable SCF energy oscillations", **kwargs
        )

    def test_dry_run_has_no_side_effects(self):
        preview = self.request(dry_run=True)
        self.assertEqual(preview[0]["task"], self.task.task_work_path)
        self.assertFalse((self.root / REQUESTS).exists())
        self.assertEqual(self.submission.machine.context.commands, [])

    def test_only_selected_current_job_is_cancelled_and_no_retry_is_submitted(self):
        self.request()
        self.assertTrue(stopped_task(self.root / self.task.task_work_path))
        self.assertFalse(stopped_task(self.root / self.other_task.task_work_path))
        self.submission.handle_unexpected_submission_state()
        self.assertEqual(self.submission.machine.context.commands, ["scancel 1"])
        entry = self.submission.recovery_ledger["tasks"][self.task.task_work_path]
        self.assertEqual(entry["status"], "discard_pending")
        self.job.submit_job.assert_not_called()
        self.other_job.submit_job.assert_not_called()
        self.submission.handle_unexpected_submission_state()
        self.assertEqual(self.submission.machine.context.commands, ["scancel 1"])
        self.job.job_state = JobStatus.terminated
        self.submission.handle_unexpected_submission_state()
        self.assertEqual(entry["status"], "discarded")
        self.assertEqual(
            read_requests(self.root)["requests"][self.task.task_work_path]["status"],
            "done",
        )
        self.assertEqual(
            (self.root / self.task.task_work_path / "POSCAR").read_text(),
            "preserved structure",
        )
        self.assertEqual(
            self.submission.recovery_ledger["tasks"][self.other_task.task_work_path][
                "status"
            ],
            "running",
        )

    def test_discard_intent_is_committed_before_scheduler_cancellation(self):
        self.request()
        original = self.submission.machine.context.block_call

        def cancel(command):
            ledger = json.loads((self.root / ".fp-recovery.json").read_text())
            self.assertEqual(
                ledger["tasks"][self.task.task_work_path]["status"], "discard_pending"
            )
            self.assertTrue(
                (self.root / self.task.task_work_path / "fp-discard.json").exists()
            )
            return original(command)

        self.submission.machine.context.block_call = cancel
        self.submission.handle_unexpected_submission_state()

    def test_never_submitted_task_is_discarded_without_cancellation(self):
        self.job.job_id = None
        self.job.job_state = JobStatus.unsubmitted
        entry = self.submission.recovery_ledger["tasks"][self.task.task_work_path]
        entry.update(job_id=None, status="pending")
        self.submission._save()
        request_discard(
            self.root,
            tasks=[self.task.task_work_path],
            reason="Known nonconverging input",
        )
        self.submission.handle_unexpected_submission_state()
        self.assertEqual(entry["status"], "discarded")
        self.assertEqual(self.submission.machine.context.commands, [])
        self.job.submit_job.assert_not_called()

    def test_controller_restart_does_not_repeat_cancel_or_resubmit(self):
        self.request()
        self.submission.handle_unexpected_submission_state()
        ledger = json.loads((self.root / ".fp-recovery.json").read_text())
        restarted = self.fixture.submission(
            [self.job, self.other_job], instrumented=True
        )
        restarted.configure_recovery(
            self.root, self.submission.recovery_options, ledger, instrumented=True
        )
        restarted.handle_unexpected_submission_state()
        self.assertEqual(restarted.machine.context.commands, [])
        self.job.submit_job.assert_not_called()
        self.job.job_state = JobStatus.terminated
        restarted.handle_unexpected_submission_state()
        self.assertEqual(
            ledger["tasks"][self.task.task_work_path]["status"], "discarded"
        )

    def test_failed_cancellation_remains_pending_and_disables_retry(self):
        self.request()
        self.submission.machine.context.block_call = Mock(
            return_value=(1, None, io.StringIO(""), io.StringIO("Permission denied"))
        )
        self.submission.handle_unexpected_submission_state()
        entry = self.submission.recovery_ledger["tasks"][self.task.task_work_path]
        self.assertEqual(entry["status"], "discard_pending")
        self.assertIn("Permission denied", entry["manual_discard"]["cancel_error"])
        self.job.submit_job.assert_not_called()

    def test_unknown_jobs_and_successful_tasks_are_rejected_before_mutation(self):
        for jobs in (["999"], ["1", "999"], ["1; scancel --user gazom"]):
            with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                request_discard(self.root, jobs=jobs, reason="SCF failed")
        self.assertFalse((self.root / REQUESTS).exists())
        self.submission.recovery_ledger["tasks"][self.task.task_work_path]["status"] = (
            "successful"
        )
        self.submission._save()
        with self.assertRaisesRegex(ValueError, "validated results"):
            self.request()
        self.assertFalse((self.root / REQUESTS).exists())

    def test_stale_input_scope_causes_no_scheduler_action(self):
        self.request()
        self.submission.recovery_ledger["tasks"][self.task.task_work_path][
            "input_digest"
        ] = "changed"
        self.submission._save()
        self.submission.handle_unexpected_submission_state()
        self.assertEqual(self.submission.machine.context.commands, [])
        self.assertEqual(
            read_requests(self.root)["requests"][self.task.task_work_path]["status"],
            "error",
        )
        self.assertFalse(stopped_task(self.root / self.task.task_work_path))

    def test_job_id_selection_remains_bound_to_task_across_requeue(self):
        self.request()
        self.job.job_id = "3"
        self.submission.handle_unexpected_submission_state()
        self.assertEqual(self.submission.machine.context.commands, ["scancel 3"])
        self.other_job.submit_job.assert_not_called()

    def test_cli_parser_requires_reason_and_supports_task_selection(self):
        from dpgen.main import main_parser

        args = main_parser().parse_args(
            [
                "fp-discard",
                str(self.root),
                "--tasks",
                self.task.task_work_path,
                "--reason",
                "SCF oscillating",
                "--dry-run",
            ]
        )
        self.assertTrue(args.dry_run)
        self.assertEqual(args.tasks, [self.task.task_work_path])

    def test_grouped_job_is_not_cancelled_for_a_single_task_request(self):
        self.request()
        self.job.job_task_list.append(self.other_task)
        self.submission.handle_unexpected_submission_state()
        self.assertEqual(self.submission.machine.context.commands, [])
        self.assertEqual(
            read_requests(self.root)["requests"][self.task.task_work_path]["status"],
            "error",
        )
