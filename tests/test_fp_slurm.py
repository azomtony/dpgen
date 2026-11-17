"""In-place Slurm FP controller: readable jobs, per-task resources, no lost results."""

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dpgen.dispatcher import fp_slurm
from dpgen.dispatcher.fp_slurm import (
    Campaign,
    SubmitRejected,
    SubmitUncertain,
    next_rung,
    options,
    queue_request,
)
from dpgen.dispatcher.fp_vasp_monitor import stopped_task

POSCAR = """Cu O
1.0
10 0 0
0 10 0
0 0 10
Cu O
{cu} {o}
Direct
"""

CONFIG = {
    "enabled": True,
    "check_interval": 1,
    "retry_delay": 0,
    "accounting_grace": 0,
    "vars": {"ngpus": 1, "time": "24:00:00"},
    "directives": [
        "--partition=gpu",
        "--gres=gpu:{ngpus}",
        "--ntasks-per-node={ngpus}",
        "--time={time}",
    ],
    "prepend": ["module load vasp/6.4-gpu"],
    "command": "mpirun -np {ngpus} vasp_std",
    "rules": [{"min_natoms": 400, "vars": {"ngpus": 2}}],
    "escalate": {"resource": {"ngpus": [2, 4]}},
}


def good(task):
    path = Path(task) / "OUTCAR"
    ok = path.exists() and path.read_text() == "GOOD"
    return ok, "converged" if ok else "not converged"


class FakeSlurm:
    def __init__(self):
        self.jobs = {}
        self.acct = {}
        self.history = []
        self.cancelled = []
        self.next_id = 100
        self.fail_next = None
        self.create_then_fail = False
        self.start_before_cancel = False

    def submit(self, script, cwd):
        assert (Path(cwd) / script).exists()
        job_id = str(self.next_id)
        self.next_id += 1
        if self.fail_next is not None and not self.create_then_fail:
            error, self.fail_next = self.fail_next, None
            raise error
        self.jobs[job_id] = {"state": "PENDING", "workdir": cwd, "name": "x"}
        self.history.append((job_id, cwd))
        if self.fail_next is not None:
            error, self.fail_next = self.fail_next, None
            raise error
        return job_id

    def queue(self):
        return {k: dict(v) for k, v in self.jobs.items()}

    def accounting(self, ids):
        return {i: self.acct[i] for i in ids if i in self.acct}

    def find(self, workdir, since):
        return [
            j
            for j, w in self.history
            if os.path.realpath(w) == os.path.realpath(workdir)
        ]

    def cancel(self, job_id, pending_only=False):
        if self.start_before_cancel and job_id in self.jobs:
            self.jobs[job_id]["state"] = "RUNNING"
        if pending_only and self.jobs.get(job_id, {}).get("state") != "PENDING":
            return  # scancel --state=PENDING ignores a job that already started
        self.cancelled.append(job_id)
        self.end(job_id, "CANCELLED by 1000")

    def run(self, job_id):
        self.jobs[job_id]["state"] = "RUNNING"

    def end(self, job_id, state, exit_code="0:0"):
        self.jobs.pop(job_id, None)
        self.acct[job_id] = {
            "job_id": job_id,
            "state": state,
            "exit_code": exit_code,
            "node": "ghpc001",
            "partition": "gpu",
            "elapsed": "00:10:00",
        }

    def job_in(self, task):
        return next(j for j, w in reversed(self.history) if w.endswith(task))


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class TestFPSlurm(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.root = self.base / "iter.000003" / "02.fp"
        self.small, self.big = "task.000.000000", "task.000.000001"
        for name, (cu, o) in ((self.small, (54, 10)), (self.big, (400, 32))):
            task = self.root / name
            task.mkdir(parents=True)
            (task / "POSCAR").write_text(POSCAR.format(cu=cu, o=o))
            for filename in ("INCAR", "POTCAR", "KPOINTS"):
                (task / filename).write_text(filename)
        self.slurm = FakeSlurm()
        self.clock = Clock()

    def _cleanup(self):
        for path in Path(self.tmp.name).rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IWUSR | stat.S_IRUSR)
        self.tmp.cleanup()

    def campaign(self, **overrides):
        config = options(dict(CONFIG, **overrides))
        return Campaign(
            self.root,
            config,
            {"fp_style": "vasp"},
            slurm=self.slurm,
            clock=self.clock,
            sleep=self.clock.sleep,
            validator=good,
        )

    def start(self, **overrides):
        campaign = self.campaign(**overrides)
        campaign.initialize(self.slurm.queue())
        campaign.step()
        return campaign

    def task(self, name):
        return self.root / name

    def entry(self, campaign, name):
        return campaign.state["tasks"][name]

    # ---------------------------------------------------------------

    def test_readable_script_names_and_natoms_rule(self):
        campaign = self.start()
        small = (self.task(self.small) / "fp.sbatch").read_text()
        big = (self.task(self.big) / "fp.sbatch").read_text()
        self.assertIn("#SBATCH --job-name=i003-000.000000-Cu54O10-64at", small)
        self.assertIn(f"#SBATCH --chdir={self.task(self.small)}", small)
        self.assertIn("#SBATCH --gres=gpu:1", small)
        self.assertIn("#SBATCH --gres=gpu:2", big)
        self.assertIn("Cu400O32-432at", big)
        policy = json.loads((self.task(self.big) / "fp-recovery.json").read_text())
        self.assertEqual(policy["command"], "mpirun -np 2 vasp_std")
        self.assertTrue((self.task(self.big) / "fp_recovery_worker.py").exists())
        self.assertEqual(len(self.slurm.history), 2)
        self.assertEqual(self.entry(campaign, self.big)["status"], "queued")
        table = (self.root / "fp-status.tsv").read_text()
        self.assertIn("Cu400O32", table)
        self.assertIn("ngpus=2", table)
        self.assertIn("submitted job", (self.root / "fp-events.log").read_text())

    def test_oom_escalates_gpus_archives_attempt_then_holds(self):
        campaign = self.start()
        task = self.task(self.big)
        job = self.slurm.job_in(self.big)
        (task / "OUTCAR").write_text("partial")
        (task / "fp-runtime.log").write_text(
            "call to cuMemAlloc returned error 2: Out of memory"
        )
        self.slurm.end(job, "FAILED", "1:0")
        campaign.step()
        self.assertIn("#SBATCH --gres=gpu:4", (task / "fp.sbatch").read_text())
        self.assertEqual((task / "attempts/001/OUTCAR").read_text(), "partial")
        self.assertIn("gpu:2", (task / "attempts/001/fp.sbatch").read_text())
        self.assertEqual(self.entry(campaign, self.big)["escalated"], {"ngpus": 4})
        self.assertIn("escalating ngpus 2→4", (self.root / "fp-events.log").read_text())

        self.slurm.end(self.slurm.job_in(self.big), "OUT_OF_MEMORY", "0:125")
        campaign.step()
        entry = self.entry(campaign, self.big)
        self.assertEqual(entry["status"], "held")
        self.assertIn("resource", entry["reason"])
        submitted = len(self.slurm.history)
        campaign.step()
        self.assertEqual(len(self.slurm.history), submitted)

    def test_user_override_on_held_task_resubmits_immediately(self):
        campaign = self.start()
        for _ in range(2):
            self.slurm.end(self.slurm.job_in(self.big), "OUT_OF_MEMORY")
            campaign.step()
        self.assertEqual(self.entry(campaign, self.big)["status"], "held")
        queue_request(self.root, "set", [self.big], vars={"ngpus": 8}, resubmit=True)
        campaign.step()
        entry = self.entry(campaign, self.big)
        self.assertEqual(entry["status"], "queued")
        self.assertEqual(entry["manual"], {"ngpus": 8})
        self.assertEqual(entry["escalated"], {})
        self.assertIn("gpu:8", (self.task(self.big) / "fp.sbatch").read_text())
        self.assertEqual(
            json.loads((self.root / "fp-requests.json").read_text())["requests"], []
        )

    def test_set_with_resubmit_cancels_running_job(self):
        campaign = self.start()
        old = self.slurm.job_in(self.small)
        self.slurm.run(old)
        campaign.step()
        (self.task(self.small) / "OUTCAR").write_text("partial")
        queue_request(self.root, "set", [self.small], vars={"ngpus": 2}, resubmit=True)
        campaign.step()  # cancels; job leaves the queue
        campaign.step()  # sees cancellation, resubmits
        self.assertEqual(self.slurm.cancelled, [old])
        new = self.slurm.job_in(self.small)
        self.assertNotEqual(new, old)
        self.assertIn("gpu:2", (self.task(self.small) / "fp.sbatch").read_text())
        self.assertEqual(
            (self.task(self.small) / "attempts/001/OUTCAR").read_text(), "partial"
        )

    def test_completed_result_is_frozen_and_never_resubmitted(self):
        campaign = self.start()
        task = self.task(self.small)
        (task / "OUTCAR").write_text("GOOD")
        self.slurm.end(self.slurm.job_in(self.small), "COMPLETED")
        campaign.step()
        self.assertEqual(self.entry(campaign, self.small)["status"], "done")
        record = json.loads((task / "fp-done.json").read_text())
        self.assertIn("OUTCAR", record["outputs"])
        self.assertFalse(os.access(task / "OUTCAR", os.W_OK))
        submitted = len(self.slurm.history)
        queue_request(self.root, "resubmit", [self.small])
        campaign.step()
        self.assertEqual(len(self.slurm.history), submitted)
        self.assertFalse(stopped_task(task))
        self.assertTrue(stopped_task(self.task(self.big)))

        (task / "OUTCAR").chmod(0o644)
        (task / "OUTCAR").write_text("tampered")
        with self.assertRaisesRegex(RuntimeError, "modified"):
            self.campaign().initialize(self.slurm.queue())

    def test_existing_valid_outputs_are_kept_without_submission(self):
        (self.task(self.small) / "OUTCAR").write_text("GOOD")
        campaign = self.start()
        self.assertEqual(self.entry(campaign, self.small)["status"], "done")
        self.assertEqual([w for _, w in self.slurm.history], [str(self.task(self.big))])

    def test_uncertain_sbatch_recovers_created_job_without_duplicate(self):
        self.slurm.fail_next = SubmitUncertain("socket timed out")
        self.slurm.create_then_fail = True
        campaign = self.start()
        submitted = len(self.slurm.history)
        self.assertTrue(self.entry(campaign, self.small).get("submitting"))
        campaign.step()
        entry = self.entry(campaign, self.small)
        self.assertNotIn("submitting", entry)
        self.assertEqual(entry["status"], "queued")
        self.assertEqual(len(self.slurm.history), submitted)

    def test_uncertain_sbatch_without_job_submits_once_more(self):
        self.slurm.fail_next = SubmitUncertain("socket timed out")
        campaign = self.start()
        self.assertEqual(len(self.slurm.history), 1)
        campaign.step()  # nothing found: clears intent
        campaign.step()  # submits again
        self.assertEqual(len(self.slurm.history), 2)
        self.assertEqual(self.entry(campaign, self.small)["status"], "queued")

    def test_rejected_sbatch_holds_with_reason(self):
        self.slurm.fail_next = SubmitRejected("Invalid partition name specified")
        campaign = self.start()
        entry = self.entry(campaign, self.small)
        self.assertEqual(entry["status"], "held")
        self.assertIn("Invalid partition", entry["reason"])

    def test_refuses_to_start_when_untracked_jobs_run_in_stage(self):
        self.slurm.jobs["7"] = {
            "state": "RUNNING",
            "workdir": str(self.task(self.small)),
            "name": "old",
        }
        with self.assertRaisesRegex(RuntimeError, "still queued or running"):
            self.campaign().initialize(self.slurm.queue())
        self.assertEqual(self.slurm.history, [])

    def test_adopts_legacy_campaign_without_resubmitting(self):
        staging = self.base / "staging" / ("a" * 40)
        for name in (self.small, self.big):
            (staging / name).mkdir(parents=True)
        (staging / self.big / "OUTCAR").write_text("GOOD")
        self.slurm.jobs["55"] = {
            "state": "RUNNING",
            "workdir": str(staging),
            "name": "a" * 40,
        }
        (self.root / ".fp-recovery.json").write_text(
            json.dumps(
                {
                    "remote_root": str(staging),
                    "submission_hash": "a" * 40,
                    "tasks": {
                        self.small: {"status": "running", "job_id": "55"},
                        self.big: {"status": "failed", "job_id": "56"},
                    },
                }
            )
        )
        campaign = self.start()
        self.assertEqual(self.slurm.history, [])
        self.assertEqual(self.entry(campaign, self.small)["status"], "running")
        self.assertEqual(self.entry(campaign, self.big)["status"], "done")
        self.assertEqual((staging / self.big / "OUTCAR").read_text(), "GOOD")

        (staging / self.small / "OUTCAR").write_text("GOOD")
        self.slurm.end("55", "COMPLETED")
        campaign.step()
        self.assertEqual(self.entry(campaign, self.small)["status"], "done")
        self.assertEqual((self.task(self.small) / "OUTCAR").read_text(), "GOOD")
        self.assertTrue((staging / self.small / "OUTCAR").exists())
        self.assertTrue(campaign.complete())

    def test_monitor_stop_is_excluded_without_retry(self):
        campaign = self.start()
        job = self.slurm.job_in(self.small)
        task = self.task(self.small)
        (task / "fp-monitor-stop.json").write_text(
            json.dumps({"job_id": job, "reason": "SCF stagnation"})
        )
        self.slurm.end(job, "FAILED", "86:0")
        campaign.step()
        self.assertEqual(self.entry(campaign, self.small)["status"], "stopped")
        self.assertTrue(stopped_task(task))

    def test_node_failure_retries_without_limit(self):
        campaign = self.start()
        for _ in range(5):
            self.slurm.end(self.slurm.job_in(self.small), "NODE_FAIL")
            campaign.step()
            self.assertEqual(self.entry(campaign, self.small)["status"], "queued")
        self.assertEqual(self.entry(campaign, self.small)["retries"]["hardware"], 5)

    def test_external_cancel_holds_for_user(self):
        campaign = self.start()
        self.slurm.end(self.slurm.job_in(self.small), "CANCELLED by 1000")
        campaign.step()
        self.assertEqual(self.entry(campaign, self.small)["status"], "held")

    def test_discard_cancels_and_excludes(self):
        campaign = self.start()
        job = self.slurm.job_in(self.big)
        queue_request(self.root, "discard", [self.big], reason="too large")
        campaign.step()
        campaign.step()
        self.assertEqual(self.slurm.cancelled, [job])
        self.assertEqual(self.entry(campaign, self.big)["status"], "discarded")

    def test_run_loop_finishes_stage(self):
        campaign = self.campaign()
        slurm = self.slurm
        original = slurm.queue

        def queue():
            for job_id, workdir in list(slurm.history):
                if job_id in slurm.jobs:
                    (Path(workdir) / "OUTCAR").write_text("GOOD")
                    slurm.end(job_id, "COMPLETED")
            return original()

        slurm.queue = queue
        counts = campaign.run()
        self.assertEqual(counts, {"done": 2})

    def test_pending_large_job_is_updated_when_rule_is_added(self):
        campaign = self.start(rules=[])
        old_big = self.slurm.job_in(self.big)
        old_small = self.slurm.job_in(self.small)
        self.assertIn("gpu:1", (self.task(self.big) / "fp.sbatch").read_text())
        # Restart dpgen with the natoms rule now in machine.json.
        campaign = self.campaign()
        campaign.initialize(self.slurm.queue())
        campaign.step()  # pending-only cancel of the stale large job
        self.assertEqual(self.slurm.cancelled, [old_big])
        campaign.step()  # replaced before it ever ran
        new_big = self.slurm.job_in(self.big)
        self.assertNotEqual(new_big, old_big)
        self.assertIn("gpu:2", (self.task(self.big) / "fp.sbatch").read_text())
        self.assertEqual(self.slurm.job_in(self.small), old_small)
        self.assertIn(
            "replacing it before it starts", (self.root / "fp-events.log").read_text()
        )
        entry = self.entry(campaign, self.big)
        self.assertEqual(entry["retries"], {})
        self.assertEqual(entry["status"], "queued")

    def test_user_override_updates_pending_job_without_resubmit_flag(self):
        campaign = self.start()
        old = self.slurm.job_in(self.big)
        queue_request(self.root, "set", [self.big], vars={"ngpus": 4})
        campaign.step()
        campaign.step()
        self.assertEqual(self.slurm.cancelled, [old])
        self.assertIn("gpu:4", (self.task(self.big) / "fp.sbatch").read_text())

    def test_update_never_kills_a_job_that_already_started(self):
        campaign = self.start(rules=[])
        job = self.slurm.job_in(self.big)
        self.slurm.start_before_cancel = True
        campaign = self.campaign()
        campaign.initialize(self.slurm.queue())
        campaign.step()
        self.assertEqual(self.slurm.cancelled, [])
        campaign.step()
        attempt = self.entry(campaign, self.big)["attempts"][-1]
        self.assertNotIn("cancel_intent", attempt)
        self.assertIn(
            "keeps its original resources", (self.root / "fp-events.log").read_text()
        )
        # It runs out of memory on 1 GPU: the next attempt uses the rule's 2,
        # without inventing an escalation above it.
        self.slurm.start_before_cancel = False
        self.slurm.end(job, "OUT_OF_MEMORY")
        campaign.step()
        self.assertIn("gpu:2", (self.task(self.big) / "fp.sbatch").read_text())
        self.assertEqual(self.entry(campaign, self.big)["escalated"], {})

    def test_update_pending_can_be_disabled(self):
        self.start(rules=[])
        campaign = self.campaign(update_pending=False)
        campaign.initialize(self.slurm.queue())
        campaign.step()
        campaign.step()
        self.assertEqual(self.slurm.cancelled, [])

    def test_adopted_pending_jobs_updated_only_when_rule_targets_them(self):
        staging = self.base / "staging" / ("a" * 40)
        staging.mkdir(parents=True)
        for job_id in ("55", "56"):
            self.slurm.jobs[job_id] = {
                "state": "PENDING",
                "workdir": str(staging),
                "name": "a" * 40,
            }
        (self.root / ".fp-recovery.json").write_text(
            json.dumps(
                {
                    "remote_root": str(staging),
                    "tasks": {
                        self.small: {"status": "waiting", "job_id": "55"},
                        self.big: {"status": "waiting", "job_id": "56"},
                    },
                }
            )
        )
        campaign = self.start()
        self.assertEqual(self.slurm.cancelled, ["56"])
        campaign.step()
        self.assertIn("gpu:2", (self.task(self.big) / "fp.sbatch").read_text())
        self.assertIn("55", self.slurm.jobs)

    def test_refuses_plain_dispatcher_jobs_staged_in_hash_directory(self):
        staging = self.base / ("b" * 40)
        (staging / self.big).mkdir(parents=True)
        (staging / self.big / "POSCAR").write_bytes(
            (self.task(self.big) / "POSCAR").read_bytes()
        )
        self.slurm.jobs["9"] = {
            "state": "PENDING",
            "workdir": str(staging),
            "name": "b" * 40,
        }
        with self.assertRaisesRegex(RuntimeError, "still queued or running"):
            self.campaign().initialize(self.slurm.queue())
        self.assertEqual(self.slurm.history, [])

    def test_worker_runs_with_dpgen_interpreter_by_default(self):
        import sys

        self.start()
        script = (self.task(self.small) / "fp.sbatch").read_text()
        self.assertIn(f"{sys.executable} fp_recovery_worker.py", script)

    # ---- dry run -------------------------------------------------------

    def snapshot(self):
        import hashlib

        result = {}
        for path in sorted(self.base.rglob("*")):
            info = path.lstat()
            digest = (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file() and not path.is_symlink()
                else None
            )
            result[str(path.relative_to(self.base))] = (
                info.st_mode,
                info.st_mtime_ns,
                digest,
            )
        return result

    def plan(self, **overrides):
        import io

        out = io.StringIO()
        code = fp_slurm.plan_stage(
            self.root,
            {"fp_style": "vasp"},
            {"fp_slurm": dict(CONFIG, **overrides)},
            slurm=self.slurm,
            out=out,
            validator=good,
        )
        return code, out.getvalue()

    def test_dry_run_new_stage_shows_scripts_and_changes_nothing(self):
        before = self.snapshot()
        code, text = self.plan()
        self.assertEqual(code, 0)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.slurm.history, [])
        self.assertIn("Mode: new campaign", text)
        self.assertIn("Would submit 2 job(s)", text)
        self.assertIn("#SBATCH --gres=gpu:2", text)
        self.assertIn("#SBATCH --job-name=i003-000.000001-Cu400O32-432at", text)
        self.assertIn("nothing was submitted", text)

    def test_dry_run_adoption_plans_updates_without_touching_anything(self):
        staging = self.base / "staging" / ("a" * 40)
        for name in (self.small, self.big):
            (staging / name).mkdir(parents=True)
            (staging / name / "POSCAR").write_bytes(
                (self.task(name) / "POSCAR").read_bytes()
            )
        (staging / self.small / "OUTCAR").write_text("GOOD")
        self.slurm.jobs["56"] = {
            "state": "PENDING",
            "workdir": str(staging),
            "name": "a" * 40,
        }
        (self.root / ".fp-recovery.json").write_text(
            json.dumps(
                {
                    "remote_root": str(staging),
                    "tasks": {
                        self.small: {"status": "running", "job_id": "55"},
                        self.big: {"status": "waiting", "job_id": "56"},
                    },
                }
            )
        )
        before = self.snapshot()
        code, text = self.plan()
        self.assertEqual(code, 0)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.slurm.cancelled, [])
        self.assertIn("adopt previous recovery controller", text)
        self.assertIn("copied OUTCAR from", text)
        self.assertIn("done (result copied from previous staging)", text)
        self.assertIn("pending job 56 requested unknown (adopted job)", text)
        self.assertIn("scancel 56 (only if still pending", text)

    def test_dry_run_reports_blocking_jobs(self):
        self.slurm.jobs["7"] = {
            "state": "RUNNING",
            "workdir": str(self.task(self.small)),
            "name": "x",
        }
        code, text = self.plan()
        self.assertEqual(code, 1)
        self.assertIn("WOULD STOP", text)
        self.assertIn("7", text)

    def test_dry_run_existing_campaign_with_rule_change_and_requests(self):
        self.start(rules=[])
        queue_request(self.root, "set", [self.small], vars={"ngpus": 4})
        before = self.snapshot()
        code, text = self.plan()
        self.assertEqual(code, 0)
        self.assertEqual(self.snapshot(), before)
        self.assertIn("resume existing campaign", text)
        self.assertIn("resources set ngpus=4", text)
        self.assertEqual(
            len(json.loads((self.root / "fp-requests.json").read_text())["requests"]), 1
        )
        self.assertEqual(text.count("only if still pending"), 2)

    def test_workflow_dry_run_follows_record_and_checks_machine_json(self):
        import io

        os.chdir(self.base)
        self.addCleanup(os.chdir, Path(__file__).resolve().parent)
        (self.base / "param.json").write_text(json.dumps({"fp_style": "vasp"}))
        machine = {
            "fp": {
                "command": "x",
                "machine": {},
                "resources": {"group_size": 1},
                "slurm": CONFIG,
            }
        }
        (self.base / "machine.json").write_text(json.dumps(machine))
        (self.base / "record.dpgen").write_text("3 6\n")
        out = io.StringIO()
        code = fp_slurm.workflow_dry_run(
            "param.json", "machine.json", out=out, slurm=self.slurm
        )
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("step 07 (run_fp)", text)
        self.assertIn("rule: min_natoms=400 -> ngpus=2", text)
        self.assertIn("Would submit 2 job(s)", text)
        self.assertFalse((self.root / "fp-campaign.json").exists())

        (self.base / "record.dpgen").write_text("3 6\n4 3\n")
        out = io.StringIO()
        fp_slurm.workflow_dry_run(
            "param.json", "machine.json", out=out, slurm=self.slurm
        )
        self.assertIn("not prepared yet", out.getvalue())

        machine["fp"]["resources"]["slurm"] = machine["fp"].pop("slurm")
        (self.base / "machine.json").write_text(json.dumps(machine))
        out = io.StringIO()
        code = fp_slurm.workflow_dry_run("param.json", "machine.json", out=out)
        self.assertEqual(code, 1)
        self.assertIn("inside fp.resources", out.getvalue())

    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, "undefined variable"):
            options(dict(CONFIG, command="mpirun -np {nranks} vasp_std"))
        with self.assertRaisesRegex(ValueError, "set by the controller"):
            options(dict(CONFIG, directives=["--chdir=/tmp"]))
        with self.assertRaisesRegex(ValueError, "no base value"):
            options(dict(CONFIG, escalate={"timeout": {"walltime": ["48:00:00"]}}))
        with self.assertRaisesRegex(ValueError, "Invalid fp.slurm"):
            options(dict(CONFIG, typo=1))

    def test_next_rung_orders_counts_times_and_memory(self):
        self.assertEqual(next_rung(1, [2, 4]), 2)
        self.assertEqual(next_rung(3, [2, 4]), 4)
        self.assertIsNone(next_rung(4, [2, 4]))
        self.assertEqual(
            next_rung("24:00:00", ["1-00:00:00", "2-00:00:00"]), "2-00:00:00"
        )
        self.assertEqual(next_rung("12:00:00", ["1-00:00:00"]), "1-00:00:00")
        self.assertEqual(next_rung("64G", ["128G", "256G"]), "128G")

    def test_cli_selection_and_done_protection(self):
        campaign = self.start()
        (self.task(self.small) / "OUTCAR").write_text("GOOD")
        self.slurm.end(self.slurm.job_in(self.small), "COMPLETED")
        campaign.step()
        state = campaign.state
        args = SimpleNamespace(
            tasks=[], jobs=[], status=[], min_natoms=400, max_natoms=None
        )
        self.assertEqual(fp_slurm.select_tasks(state, args)[0], [self.big])
        args = SimpleNamespace(
            tasks=["000.000000"], jobs=[], status=[], min_natoms=None, max_natoms=None
        )
        with self.assertRaises(SystemExit):
            fp_slurm.select_tasks(state, args)

    def test_requires_fp_slurm_when_campaign_exists(self):
        self.start()
        with self.assertRaisesRegex(RuntimeError, "not enabled"):
            fp_slurm.campaign_requested(self.root, {})
        self.assertTrue(
            fp_slurm.campaign_requested(self.root, {"fp_slurm": {"enabled": True}})
        )


if __name__ == "__main__":
    unittest.main()
