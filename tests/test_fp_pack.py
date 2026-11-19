"""Packing FP single points into long allocations: controller + allocation worker."""

import io
import json
import os
import signal
import stat
import tempfile
import unittest
from pathlib import Path

from dpgen.dispatcher import fp_pack, fp_slurm
from dpgen.dispatcher.fp_pilot import Pilot, visible_gpus
from dpgen.dispatcher.fp_slurm import Campaign, options, queue_request

POSCAR = "Cu O\n1.0\n10 0 0\n0 10 0\n0 0 10\nCu O\n{cu} {o}\nDirect\n"

PACK = {
    "enabled": True,
    "gpus": 4,
    "time": "24:00:00",
    "max_allocations": 1,
    "directives": [
        "--partition={partition}",
        "--nodes=1",
        "--ntasks-per-node={gpus}",
        "--gres=gpu:{gpus}",
        "--time={time}",
    ],
    "default_runtime": "01:00:00",
    "start_margin": 600,
}
CONFIG = {
    "enabled": True,
    "check_interval": 1,
    "retry_delay": 0,
    "accounting_grace": 0,
    "vars": {"ngpus": 1, "time": "24:00:00", "partition": "gpu"},
    "directives": ["--partition={partition}", "--gres=gpu:{ngpus}"],
    "prepend": ["source vasp.sh"],
    "command": "mpirun --bind-to none -np {ngpus} vasp_std",
    "rules": [{"min_natoms": 400, "vars": {"ngpus": 2}}],
    "escalate": {"resource": {"ngpus": [2, 4]}},
    "pack": PACK,
}


def good(task):
    path = Path(task) / "OUTCAR"
    ok = path.exists() and path.read_text() == "GOOD"
    return ok, "converged" if ok else "not converged"


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeSlurm:
    def __init__(self):
        self.jobs, self.acct, self.history, self.cancelled = {}, {}, [], []
        self.next_id = 500

    def submit(self, script, cwd):
        job_id = str(self.next_id)
        self.next_id += 1
        self.jobs[job_id] = {"state": "PENDING", "workdir": cwd, "name": script}
        self.history.append((job_id, cwd))
        return job_id

    def queue(self):
        return {k: dict(v) for k, v in self.jobs.items()}

    def accounting(self, ids):
        return {i: self.acct[i] for i in ids if i in self.acct}

    def find(self, workdir, since):
        return [j for j, w in self.history if w == str(workdir)]

    def cancel(self, job_id, pending_only=False):
        if pending_only and self.jobs.get(job_id, {}).get("state") != "PENDING":
            return
        self.cancelled.append(job_id)
        self.end(job_id, "CANCELLED")

    def start(self, job_id):
        self.jobs[job_id]["state"] = "RUNNING"

    def end(self, job_id, state="COMPLETED"):
        self.jobs.pop(job_id, None)
        self.acct[job_id] = {"job_id": job_id, "state": state, "exit_code": "0:0"}


class FakeProcess:
    def __init__(self, pid, cwd, env):
        self.pid, self.cwd, self.env, self.code = pid, Path(cwd), env, None

    def poll(self):
        return self.code


class FakePopen:
    def __init__(self):
        self.by_task, self.by_pid = {}, {}

    def __call__(self, args, cwd, env, stdout, stderr, start_new_session):
        process = FakeProcess(2000 + len(self.by_pid), cwd, env)
        self.by_task[Path(cwd).name] = process
        self.by_pid[process.pid] = process
        return process

    def killpg(self, pid, sig):
        self.by_pid[pid].code = -int(sig)


class TestPack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.root = self.base / "iter.000004" / "02.fp"
        self.small, self.big = "task.000.000000", "task.000.000001"
        for name, (cu, o) in ((self.small, (54, 10)), (self.big, (400, 32))):
            task = self.root / name
            task.mkdir(parents=True)
            (task / "POSCAR").write_text(POSCAR.format(cu=cu, o=o))
            for filename in ("INCAR", "POTCAR", "KPOINTS"):
                (task / filename).write_text(filename)
        self.slurm = FakeSlurm()
        self.clock = Clock()
        self.popen = FakePopen()
        self.pilots = {}

    def _cleanup(self):
        for pilot in self.pilots.values():
            for info in pilot.busy.values():
                info["log"].close()
        for path in Path(self.tmp.name).rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IWUSR | stat.S_IRUSR)
        self.tmp.cleanup()

    def campaign(self, **overrides):
        pack = dict(PACK, **overrides.pop("pack", {}))
        config = options(dict(CONFIG, pack=pack, **overrides))
        campaign = Campaign(
            self.root,
            config,
            {"fp_style": "vasp"},
            slurm=self.slurm,
            clock=self.clock,
            sleep=self.clock.sleep,
            validator=good,
        )
        campaign.initialize(self.slurm.queue())
        return campaign

    def entry(self, campaign, name):
        return campaign.state["tasks"][name]

    def alloc(self, campaign, alloc_id="A01"):
        return campaign.state["allocations"][alloc_id]

    def start_allocation(self, campaign, alloc_id="A01", time_limit=86400):
        alloc = self.alloc(campaign, alloc_id)
        self.slurm.start(alloc["job_id"])
        pilot = Pilot(
            alloc["dir"],
            4,
            time_limit,
            900,
            python="python3",
            clock=self.clock,
            sleep=self.clock.sleep,
            popen=self.popen,
            environment={
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "SLURM_JOB_ID": alloc["job_id"],
            },
            killpg=self.popen.killpg,
        )
        pilot.heartbeat()
        self.pilots[alloc_id] = pilot
        return pilot

    @staticmethod
    def tick(pilot):
        pilot.heartbeat()
        pilot.handle_cancels()
        pilot.poll()
        pilot.start_tasks()

    def complete(self, pilot, name, code=0, outcar="GOOD", log=""):
        task = self.root / name
        os.chmod(task, 0o755)
        if outcar is not None:
            (task / "OUTCAR").write_text(outcar)
        (task / "fp-runtime.log").write_text(log)
        (task / "fp-attempt.json").write_text(
            json.dumps(
                {
                    "job_id": pilot.job_id,
                    "started": self.clock.now - 3000,
                    "finished": self.clock.now,
                    "exit_code": code,
                }
            )
        )
        self.popen.by_task[name].code = code

    # ------------------------------------------------------------------

    def test_requests_allocation_and_assigns_only_after_it_starts(self):
        campaign = self.campaign()
        campaign.step()
        alloc = self.alloc(campaign)
        self.assertEqual(alloc["status"], "queued")
        self.assertEqual(len(self.slurm.history), 1)
        script = (Path(alloc["dir"]) / "alloc.sbatch").read_text()
        self.assertIn("#SBATCH --gres=gpu:4", script)
        self.assertIn("#SBATCH --signal=B:USR1@300", script)
        self.assertIn("exec ", script)
        self.assertIn("fp_pilot.py --gpus 4 --time-limit 86400", script)
        self.assertTrue((Path(alloc["dir"]) / "fp_pilot.py").exists())
        self.assertEqual(self.entry(campaign, self.small)["status"], "pending")
        campaign.step()
        self.assertEqual(len(self.slurm.history), 1)  # still one allocation

        pilot = self.start_allocation(campaign)
        campaign.step()
        for name in (self.small, self.big):
            self.assertEqual(self.entry(campaign, name)["status"], "queued")
        self.tick(pilot)
        campaign.step()
        big = self.entry(campaign, self.big)["attempts"][-1]
        small = self.entry(campaign, self.small)["attempts"][-1]
        self.assertEqual(self.entry(campaign, self.big)["status"], "running")
        self.assertEqual(big["gpus"], ["0", "1"])  # longest first, 2 GPUs (rule)
        self.assertEqual(small["gpus"], ["2"])
        self.assertEqual(
            self.popen.by_task[self.big].env["CUDA_VISIBLE_DEVICES"], "0,1"
        )
        policy = json.loads((self.root / self.big / "fp-recovery.json").read_text())
        self.assertEqual(policy["command"], "mpirun --bind-to none -np 2 vasp_std")
        self.assertEqual(len(self.slurm.history), 1)

    def test_results_collected_per_task_and_oom_retried_in_same_allocation(self):
        campaign = self.campaign()
        campaign.step()
        pilot = self.start_allocation(campaign)
        campaign.step()
        self.tick(pilot)
        campaign.step()

        self.complete(pilot, self.small)
        self.complete(
            pilot, self.big, code=1, outcar="partial", log="CUDA error: out of memory"
        )
        self.tick(pilot)
        campaign.step()  # small done; big escalates 2 -> 4 GPUs and is reassigned
        self.assertEqual(self.entry(campaign, self.small)["status"], "done")
        big = self.entry(campaign, self.big)
        self.assertEqual(big["escalated"], {"ngpus": 4})
        self.assertEqual(big["status"], "queued")
        self.assertEqual(big["attempts"][-1]["allocation"], "A01")
        self.assertEqual(
            (self.root / self.big / "attempts/001/OUTCAR").read_text(), "partial"
        )
        self.tick(pilot)
        self.assertEqual(
            self.popen.by_task[self.big].env["CUDA_VISIBLE_DEVICES"], "0,1,2,3"
        )
        history = (self.base / fp_pack.RUNTIME_HISTORY).read_text().splitlines()
        self.assertEqual(json.loads(history[0])["natoms"], 64)
        self.assertEqual(len(self.slurm.history), 1)

    def test_preemption_requeues_without_using_a_retry(self):
        campaign = self.campaign()
        campaign.step()
        pilot = self.start_allocation(campaign)
        campaign.step()
        self.tick(pilot)
        campaign.step()
        self.clock.now += 1800
        pilot.stop_all(signal.SIGTERM)
        self.slurm.end(self.alloc(campaign)["job_id"], "PREEMPTED")
        campaign.step()
        for name in (self.small, self.big):
            entry = self.entry(campaign, name)
            self.assertEqual(entry["status"], "pending")
            self.assertEqual(entry["retries"], {})
            self.assertGreaterEqual(entry["min_runtime"], 1800)
        self.assertEqual(self.alloc(campaign)["status"], "ended")
        self.assertEqual(self.alloc(campaign, "A02")["status"], "queued")

    def test_tasks_not_started_go_back_when_allocation_ends(self):
        campaign = self.campaign()
        campaign.step()
        self.start_allocation(campaign)
        campaign.step()  # assigned, worker never picks them up
        self.slurm.end(self.alloc(campaign)["job_id"], "TIMEOUT")
        campaign.step()
        for name in (self.small, self.big):
            entry = self.entry(campaign, name)
            self.assertEqual(entry["attempts"][-1]["category"], "not_started")
            self.assertIn(entry["status"], ("pending", "queued"))

    def test_task_must_fit_remaining_time(self):
        campaign = self.campaign(pack={"max_allocations": 2})
        campaign.step()
        self.start_allocation(campaign, time_limit=3600)  # 1 h left < estimate+margin
        campaign.step()
        self.assertEqual(self.entry(campaign, self.small)["status"], "pending")
        self.assertIn("A02", campaign.state["allocations"])

    def test_drains_idle_allocation_and_cancels_unneeded_pending_one(self):
        campaign = self.campaign(pack={"max_allocations": 2})
        campaign.step()
        pilot = self.start_allocation(campaign)
        campaign.step()
        self.tick(pilot)
        campaign.step()
        campaign.state["allocations"]["A02"] = dict(
            self.alloc(campaign),
            id="A02",
            status="queued",
            assigned={},
            job_id=self.slurm.submit("alloc.sbatch", "/elsewhere"),
            dir=str(self.root / "allocations" / "A02"),
        )
        self.complete(pilot, self.small)
        self.complete(pilot, self.big)
        self.tick(pilot)
        campaign.step()
        self.assertTrue(campaign.complete())
        self.assertTrue((Path(self.alloc(campaign)["dir"]) / "drain").exists())
        self.assertIn(self.alloc(campaign, "A02")["job_id"], self.slurm.cancelled)
        self.assertNotIn(self.alloc(campaign)["job_id"], self.slurm.cancelled)
        pilot.last_activity = self.clock.now
        self.assertEqual(pilot.run(interval=0, max_loops=3), 0)
        self.assertEqual(pilot.exit_reason, "drained by controller")

    def test_resubmit_stops_only_that_task_never_the_allocation(self):
        campaign = self.campaign()
        campaign.step()
        pilot = self.start_allocation(campaign)
        campaign.step()
        self.tick(pilot)
        campaign.step()
        queue_request(self.root, "set", [self.small], vars={"ngpus": 2}, resubmit=True)
        campaign.step()
        self.assertEqual(self.slurm.cancelled, [])
        self.assertTrue(
            (
                Path(self.alloc(campaign)["dir"]) / "cancel" / f"{self.small}.json"
            ).exists()
        )
        self.tick(pilot)  # worker stops the task
        campaign.step()  # reassigned with 2 GPUs
        entry = self.entry(campaign, self.small)
        self.assertEqual(entry["attempts"][-2]["category"], "cancelled")
        self.assertEqual(entry["attempts"][-1]["vars"]["ngpus"], 2)
        self.assertIsNone(self.popen.by_task[self.big].code)  # untouched

    def test_dry_run_shows_allocation_and_estimates_writes_nothing(self):
        before = sorted(str(p) for p in self.base.rglob("*"))
        out = io.StringIO()
        config = dict(CONFIG, command="mpirun -np {ngpus} vasp_std")
        code = fp_slurm.plan_stage(
            self.root,
            {"fp_style": "vasp"},
            {"fp_slurm": config},
            slurm=self.slurm,
            out=out,
            validator=good,
        )
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertEqual(sorted(str(p) for p in self.base.rglob("*")), before)
        self.assertEqual(self.slurm.history, [])
        self.assertIn("alloc.sbatch that would be written for allocation A01", text)
        self.assertIn("2 task(s) wait for a running allocation", text)
        self.assertIn("default_runtime", text)
        self.assertIn("--bind-to", text)

    def test_runtime_model(self):
        path = self.base / "history.jsonl"
        model = fp_pack.RuntimeModel(path, 3600, site="delta")
        self.assertEqual(model.estimate(400, 1)[0], 3600)
        model.record(200, 1, 1000, "a")
        value, basis, _ = model.estimate(400, 1)
        self.assertAlmostEqual(value, 1000 * 2**2.5, places=3)
        self.assertIn("scaled", basis)
        model.record(300, 1, 2000, "b")
        model.record(400, 1, 3500, "c")
        value, basis, _ = model.estimate(400, 1)
        self.assertIn("fit of 3", basis)
        self.assertTrue(2500 < value < 4500)
        self.assertEqual(len(fp_pack.RuntimeModel(path, 1, site="delta").rows), 3)
        self.assertTrue(model.estimate(400, 1)[2])
        self.assertFalse(model.estimate(2000, 1)[2])  # far outside 200-400 atoms
        self.assertIn("other GPU counts", model.estimate(400, 2)[1])

    def test_uncertain_estimate_is_capped_but_firm_one_holds(self):
        campaign = self.campaign()
        campaign.runtime.record(64, 1, 3000, "seed")  # one small run only
        value, basis, firm = campaign._pack().estimate(self.big)
        self.assertFalse(firm)
        self.assertLessEqual(value, 86400 - 600)
        self.assertIn("capped", basis)
        self.entry(campaign, self.big)["min_runtime"] = 90000  # really ran > 24 h
        campaign.step()
        self.assertEqual(self.entry(campaign, self.big)["status"], "held")
        self.assertIn(
            "exceeds the allocation time", self.entry(campaign, self.big)["reason"]
        )

    def test_pilot_rules(self):
        self.assertEqual(visible_gpus({"CUDA_VISIBLE_DEVICES": "2,3"}, 2), ["2", "3"])
        self.assertEqual(visible_gpus({}, 2), ["0", "1"])
        root = self.base / "alloc"
        pilot = Pilot(
            root,
            2,
            3600,
            900,
            clock=self.clock,
            popen=self.popen,
            environment={"CUDA_VISIBLE_DEVICES": "0,1"},
            killpg=self.popen.killpg,
        )
        self.pilots["unit"] = pilot
        task = self.root / self.small
        for name, need, estimate in (("big", 4, 60), ("long", 1, 7200), ("ok", 2, 60)):
            (root / "inbox" / f"{name}.json").write_text(
                json.dumps(
                    {
                        "name": name,
                        "task_dir": str(task),
                        "ngpus": need,
                        "estimate": estimate,
                    }
                )
            )
        pilot.start_tasks()
        finished = {
            p.stem: json.loads(p.read_text())
            for p in (root / "finished").glob("*.json")
        }
        self.assertIn("needs 4 GPUs", finished["big"]["reason"])
        self.assertIn("not enough time", finished["long"]["reason"])
        self.assertEqual(list(pilot.busy), ["ok"])

    def test_pack_options(self):
        with self.assertRaisesRegex(ValueError, "undefined variable"):
            options(dict(CONFIG, pack=dict(PACK, directives=["--account={acct}"])))
        with self.assertRaisesRegex(ValueError, "set by the controller"):
            options(dict(CONFIG, pack=dict(PACK, directives=["--signal=B:USR1@60"])))
        self.assertEqual(fp_pack.seconds("1-02:00:00"), 93600)
        self.assertEqual(fp_pack.seconds("90:00"), 5400)
        self.assertEqual(options(CONFIG)["pack"]["time_seconds"], 86400)


if __name__ == "__main__":
    unittest.main()
