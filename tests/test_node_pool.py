"""Run-wide node pool: training/exploration through batch_type Pilot, FP shared."""

import json
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path

from dpgen.dispatcher import node_pool
from dpgen.dispatcher.Dispatcher import make_submission
from dpgen.dispatcher.fp_pilot import Pilot as Worker
from dpgen.dispatcher.fp_slurm import Campaign, options
from dpgen.dispatcher.pilot_machine import Pilot

POOL = {
    "gpus": 4,
    "time": "24:00:00",
    "max_allocations": {"train": 1, "model_devi": 1, "fp": 2},
    "directives": ["--partition={partition}", "--nodes=1", "--gres=gpu:{gpus}"],
    "vars": {"partition": "gpu"},
    "prepend": ["module load cuda"],
    "runtime": {"train": "04:00:00"},
}


class FakeSlurm:
    def __init__(self):
        self.jobs, self.history, self.cancelled = {}, [], []
        self.next_id = 700

    def submit(self, script, cwd):
        job_id = str(self.next_id)
        self.next_id += 1
        self.jobs[job_id] = {"state": "PENDING", "workdir": cwd, "name": script}
        self.history.append((job_id, cwd))
        return job_id

    def queue(self):
        return {k: dict(v) for k, v in self.jobs.items()}

    def accounting(self, ids):
        return {}

    def find(self, workdir, since):
        return [j for j, w in self.history if w == str(workdir)]

    def cancel(self, job_id, pending_only=False):
        self.cancelled.append(job_id)
        self.jobs.pop(job_id, None)


class TestNodePool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._cleanup)
        self.work = Path(self.tmp.name).resolve()
        self.cwd = os.getcwd()
        os.chdir(self.work)
        self.addCleanup(os.chdir, self.cwd)
        self.slurm = FakeSlurm()
        Pilot.slurm_factory = lambda: self.slurm
        self.addCleanup(setattr, Pilot, "slurm_factory", None)
        node_pool.configure(self.work, {"pool": POOL})
        self.workers = []

    def _cleanup(self):
        for worker in self.workers:
            for info in worker.busy.values():
                info["log"].close()
        for path in self.work.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IWUSR | stat.S_IRUSR)
        self.tmp.cleanup()

    def pool(self):
        root = node_pool.pool_root(self.work)
        return node_pool.NodePool(root, node_pool.load_config(root), self.slurm)

    def stage(self, name, count):
        stage = self.work / "iter.000002" / name
        tasks = []
        for index in range(count):
            task = stage / f"{index:03d}"
            task.mkdir(parents=True)
            tasks.append(str(task.relative_to(stage)))
        return stage, tasks

    def submission(self, stage, tasks, command):
        return make_submission(
            {
                "batch_type": "Pilot",
                "context_type": "LazyLocalContext",
                "local_root": "./",
            },
            {
                "number_node": 1,
                "cpu_per_node": 1,
                "gpu_per_node": 1,
                "queue_name": "pool",
                "group_size": 1,
            },
            [command],
            str(stage.relative_to(self.work)),
            tasks,
            1,
            [],
            [],
            [],
            "job.log",
            "job.log",
        )

    def start_worker(self, alloc_id="A01"):
        alloc = self.pool().allocs[alloc_id]
        self.slurm.jobs[alloc["job_id"]]["state"] = "RUNNING"
        worker = Worker(
            alloc["dir"],
            4,
            86400,
            1800,
            python="python3",
            environment=dict(
                os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3", SLURM_JOB_ID=alloc["job_id"]
            ),
        )
        worker.heartbeat()
        self.workers.append(worker)
        return worker

    def settle(self, submission, worker):
        for _ in range(200):
            worker.heartbeat()
            worker.poll()
            worker.start_tasks()
            self.pool().step_items(self.slurm.queue())
            submission.update_submission_state()
            if submission.check_all_finished():
                return
            time.sleep(0.05)
        self.fail("jobs did not finish")

    def test_training_runs_four_jobs_side_by_side_on_one_allocation(self):
        stage, tasks = self.stage("00.train", 4)
        submission = self.submission(
            stage, tasks, 'echo "$CUDA_VISIBLE_DEVICES" > gpu.txt'
        )
        submission.run_submission(exit_on_submit=True, clean=False)
        pool = self.pool()
        self.assertEqual(len(pool.items), 4)
        self.assertTrue(all(i["status"] == "waiting" for i in pool.items.values()))
        self.assertEqual(len(self.slurm.history), 1)  # one allocation for 4 jobs
        script = (Path(pool.allocs["A01"]["dir"]) / "alloc.sbatch").read_text()
        self.assertIn("#SBATCH --gres=gpu:4", script)
        self.assertIn("module load cuda", script)
        self.assertIn("for train", (pool.root / "pool-events.log").read_text())

        worker = self.start_worker()
        self.pool().step_items(self.slurm.queue())
        self.assertEqual(
            sorted(len(i) for i in [self.pool().allocs["A01"]["assigned"]]), [4]
        )
        self.settle(submission, worker)
        live = (self.work / "dpgen-live.log").read_text()
        self.assertIn("pool | 4 train running | nodes A01 4/4 GPU", live)
        self.assertIn(
            "node A01 (job 700) started", (self.work / "dpgen-summary.log").read_text()
        )
        gpus = sorted((stage / t / "gpu.txt").read_text().strip() for t in tasks)
        self.assertEqual(gpus, ["0", "1", "2", "3"])
        pool = self.pool()
        self.assertEqual({i["status"] for i in pool.items.values()}, {"finished"})
        self.assertEqual(len(pool.state["history"]["train"]), 4)
        self.assertEqual(pool.allocs["A01"]["assigned"], {})

        # Exploration of the same iteration reuses the node: no new sbatch.
        explore, etasks = self.stage("01.model_devi", 2)
        second = self.submission(explore, etasks, "echo explored > out.txt")
        second.run_submission(exit_on_submit=True, clean=False)
        self.settle(second, worker)
        self.assertEqual(len(self.slurm.history), 1)
        self.assertTrue((explore / "000" / "out.txt").exists())

    def test_command_envs_give_freeze_its_own_environment(self):
        stage, tasks = self.stage("00.train", 1)
        (self.work / "dp.sh").write_text("export FOO=train\n")
        (self.work / "dp-freeze.sh").write_text("export BAR=freeze\n")
        submission = make_submission(
            {
                "batch_type": "Pilot",
                "context_type": "LazyLocalContext",
                "local_root": "./",
            },
            {
                "number_node": 1,
                "cpu_per_node": 1,
                "gpu_per_node": 1,
                "queue_name": "pool",
                "group_size": 1,
                "source_list": [str(self.work / "dp.sh")],
                "envs": {"OMP_NUM_THREADS": "1"},
                "command_envs": {
                    "freeze": {"source_list": [str(self.work / "dp-freeze.sh")]}
                },
            },
            [
                'echo "$FOO-${BAR:-none}-$OMP_NUM_THREADS" > train.txt',
                'echo "${FOO:-none}-$BAR-$OMP_NUM_THREADS" > freeze.txt',
            ],
            str(stage.relative_to(self.work)),
            tasks,
            1,
            [],
            [],
            [],
            "train.log",
            "train.log",
        )
        submission.run_submission(exit_on_submit=True, clean=False)
        worker = self.start_worker()
        self.settle(submission, worker)
        task = stage / tasks[0]
        self.assertEqual((task / "train.txt").read_text().strip(), "train-none-1")
        self.assertEqual((task / "freeze.txt").read_text().strip(), "none-freeze-1")

    def test_failed_job_is_terminated_and_resubmitted_by_dispatcher(self):
        stage, tasks = self.stage("00.train", 1)
        flag = self.work / "fail-once"
        flag.write_text("x")
        submission = self.submission(
            stage,
            tasks,
            f"if [ -e {flag} ]; then rm {flag}; exit 3; fi; echo ok > ok.txt",
        )
        submission.run_submission(exit_on_submit=True, clean=False)
        worker = self.start_worker()
        for _ in range(200):
            worker.poll()
            worker.start_tasks()
            self.pool().step_items(self.slurm.queue())
            submission.update_submission_state()
            submission.handle_unexpected_submission_state()
            if submission.check_all_finished():
                break
            time.sleep(0.05)
        self.assertTrue((stage / "000" / "ok.txt").exists())
        keys = sorted(self.pool().items)
        self.assertEqual([k.rsplit(".", 1)[1] for k in keys], ["1", "2"])
        self.assertEqual(self.pool().items[keys[0]]["exit_code"], 1)

    def test_job_waits_when_allocation_ends_before_it_starts(self):
        stage, tasks = self.stage("01.model_devi", 1)
        submission = self.submission(stage, tasks, "true")
        submission.run_submission(exit_on_submit=True, clean=False)
        self.start_worker()
        pool = self.pool()
        pool.step_items(self.slurm.queue())
        key = next(iter(pool.items))
        self.assertEqual(pool.items[key]["status"], "assigned")
        self.slurm.jobs.clear()  # allocation gone before the worker started it
        pool = self.pool()
        pool.step_items(self.slurm.queue())
        self.assertEqual(pool.items[key]["status"], "waiting")
        self.assertEqual(pool.allocs["A01"]["status"], "ended")
        self.assertIn("A02", pool.allocs)  # a new node is requested for it

    def test_fp_shares_the_run_pool(self):
        stage, tasks = self.stage("00.train", 1)
        submission = self.submission(stage, tasks, "true")
        submission.run_submission(exit_on_submit=True, clean=False)
        worker = self.start_worker()
        self.settle(submission, worker)

        root = self.work / "iter.000002" / "02.fp"
        task = root / "task.000.000000"
        task.mkdir(parents=True)
        (task / "POSCAR").write_text("Cu\n1.0\n10 0 0\n0 10 0\n0 0 10\nCu\n4\nDirect\n")
        for name in ("INCAR", "POTCAR", "KPOINTS"):
            (task / name).write_text(name)
        config = options(
            {
                "enabled": True,
                "accounting_grace": 0,
                "vars": {"ngpus": 1},
                "prepend": ["source vasp.sh"],
                "command": "mpirun --bind-to none -np {ngpus} vasp_std",
                "pack": {"enabled": True, "shared": True},
            }
        )
        config["pool"] = node_pool.pool_options(POOL)
        campaign = Campaign(root, config, {"fp_style": "vasp"}, slurm=self.slurm)
        campaign.initialize(self.slurm.queue())
        campaign.step()
        entry = campaign.state["tasks"]["task.000.000000"]
        self.assertEqual(entry["attempts"][-1]["allocation"], "A01")
        self.assertEqual(len(self.slurm.history), 1)  # same node, no new request
        launcher = (task / "fp-pilot.sh").read_text()
        self.assertIn("source vasp.sh", launcher)
        self.assertIn("exec ", launcher)
        self.assertNotIn("#SBATCH", launcher)
        assigned = self.pool().allocs["A01"]["assigned"]
        self.assertEqual(assigned["i002.task.000.000000"]["owner"], "fp:i002")
        # The shared pool keeps one node for the next stage.
        self.assertEqual(campaign._pack().pool.config["keep"], 1)

    def test_successor_requested_when_node_is_about_to_end(self):
        stage, tasks = self.stage("00.train", 1)
        submission = self.submission(stage, tasks, "true")
        submission.run_submission(exit_on_submit=True, clean=False)
        self.start_worker()
        pool = self.pool()
        alloc = pool.allocs["A01"]
        # 2 h left: a 4 h training job (pool.runtime.train) cannot start there.
        status = json.loads((Path(alloc["dir"]) / "status.json").read_text())
        status["deadline"] = time.time() + 7200
        (Path(alloc["dir"]) / "status.json").write_text(json.dumps(status))
        pool.step_items(self.slurm.queue())
        pool = self.pool()
        self.assertEqual(next(iter(pool.items.values()))["status"], "waiting")
        self.assertEqual(sorted(pool.allocs), ["A01", "A02"])

    def test_status_command(self):
        import io
        from types import SimpleNamespace

        stage, tasks = self.stage("00.train", 2)
        submission = self.submission(stage, tasks, "true")
        submission.run_submission(exit_on_submit=True, clean=False)
        out = io.StringIO()
        node_pool.command(
            SimpleNamespace(
                WORKDIR=str(self.work), all=False, drain=False, retry=False
            ),
            out,
        )
        text = out.getvalue()
        self.assertIn("A01", text)
        self.assertIn("i002 train 000", text)
        self.assertIn("waiting=2", text)

    def node(self, pool, hours_left):
        """A started allocation with ``hours_left`` on its clock."""
        pool.submit_allocation("test", "fp")
        alloc = pool.allocs[sorted(pool.allocs)[-1]]
        self.slurm.jobs[alloc["job_id"]]["state"] = "RUNNING"
        now = time.time()
        (Path(alloc["dir"]) / "status.json").write_text(
            json.dumps(
                {"started": now, "deadline": now + hours_left * 3600, "time": now}
            )
        )
        return alloc

    def test_after_fp_one_node_is_kept_for_training_and_the_others_released(self):
        pool = self.pool()
        short = self.node(pool, 10)  # A01
        long = self.node(pool, 20)  # A02: most time left, kept
        pool.submit_allocation("test", "fp")  # A03 still in the queue
        pending = pool.allocs["A03"]["job_id"]
        pool.step_items(self.slurm.queue())  # FP finished: nothing waiting
        self.assertTrue((Path(short["dir"]) / "drain").exists())
        self.assertFalse((Path(long["dir"]) / "drain").exists())
        self.assertIn(pending, self.slurm.cancelled)
        events = (pool.root / "pool-events.log").read_text()
        self.assertIn("another node is kept for the next stage", events)

        # Training of the next iteration lands on the kept node.
        stage, tasks = self.stage("00.train", 4)
        submission = self.submission(stage, tasks, "true")
        submission.run_submission(exit_on_submit=True, clean=False)
        pool = self.pool()
        pool.step_items(self.slurm.queue())
        self.assertEqual({i["allocation"] for i in pool.items.values()}, {long["id"]})

    def test_kept_node_is_renewed_before_it_ends(self):
        config = dict(POOL, renew_before="04:00:00")
        node_pool.configure(self.work, {"pool": config})
        pool = self.pool()
        alloc = self.node(pool, 3)
        pool.step_items(self.slurm.queue())
        self.assertEqual(sorted(pool.allocs), ["A01"])  # idle: nothing to renew for
        pool.enqueue(
            "i002.model_devi.x.1",
            kind="model_devi",
            cwd=self.work,
            argv=["true"],
            ngpus=1,
        )
        pool.step_items(self.slurm.queue())
        self.assertEqual(pool.items["i002.model_devi.x.1"]["allocation"], alloc["id"])
        self.assertEqual(sorted(pool.allocs), ["A01", "A02"])
        self.assertIn("renewal: A01 ends in", pool.allocs["A02"]["reason"])
        pool.step_items(self.slurm.queue())
        self.assertEqual(sorted(pool.allocs), ["A01", "A02"])  # only once

    def test_idle_node_too_close_to_its_end_is_released_at_once(self):
        pool = self.pool()
        old = self.node(pool, 2)  # 2 h left; training needs 4 h (pool.runtime)
        pool.enqueue(
            "i002.train.x.1", kind="train", cwd=self.work, argv=["true"], ngpus=1
        )
        pool.step_items(self.slurm.queue())
        self.assertEqual(pool.items["i002.train.x.1"]["status"], "waiting")
        self.assertTrue((Path(old["dir"]) / "drain").exists())
        self.assertIn("A02", pool.allocs)  # its successor takes the job

    def test_node_that_just_started_counts_and_is_not_drained(self):
        pool = self.pool()
        pool.enqueue(
            "i002.train.x.1", kind="train", cwd=self.work, argv=["true"], ngpus=8
        )
        pool.items["i002.train.x.1"]["ngpus"] = 4
        for _ in range(3):
            pool.enqueue(
                f"i002.train.y.{_}", kind="train", cwd=self.work, argv=["true"], ngpus=4
            )
        pool.step_items(self.slurm.queue())
        alloc = pool.allocs["A01"]
        self.slurm.jobs[alloc["job_id"]]["state"] = "RUNNING"  # no heartbeat yet
        for _ in range(3):
            pool.step_items(self.slurm.queue())
        self.assertEqual(sorted(pool.allocs), ["A01"])  # limit train=1 respected
        self.assertFalse((Path(alloc["dir"]) / "drain").exists())

    def test_repeated_failed_starts_stop_requests(self):
        pool = self.pool()
        pool.enqueue("i002.fp.x.1", kind="fp", cwd=self.work, argv=["true"], ngpus=1)
        for expected in ("A01", "A02"):
            pool.step_items(self.slurm.queue())
            alloc = pool.allocs[expected]
            (Path(alloc["dir"]) / f"slurm-{alloc['job_id']}.out").write_text(
                "python3.12: No such file or directory\n"
            )
            del self.slurm.jobs[alloc["job_id"]]  # the job died at once
            pool.step_items(self.slurm.queue())
        self.assertIn("ended before the pilot started", pool.state["halted"])
        count = len(pool.allocs)
        pool.step_items(self.slurm.queue())
        self.assertEqual(len(pool.allocs), count)  # no more requests
        import io
        from types import SimpleNamespace

        out = io.StringIO()
        args = SimpleNamespace(
            WORKDIR=str(self.work), all=False, drain=False, retry=False
        )
        node_pool.command(args, out)
        self.assertIn("NODE REQUESTS STOPPED", out.getvalue())
        node_pool.command(SimpleNamespace(**{**vars(args), "retry": True}), out)
        self.assertNotIn("halted", self.pool().state)

    def test_pool_options(self):
        config = node_pool.pool_options(POOL)
        self.assertEqual(config["time_seconds"], 86400)
        self.assertEqual(config["runtime_seconds"]["train"], 14400)
        pool = node_pool.NodePool(self.work / "x", config, self.slurm)
        self.assertEqual(pool.limit("fp"), 2)
        self.assertEqual(pool.limit("other"), 1)
        self.assertEqual(pool.estimate("train")[0], 14400)
        self.assertEqual(pool.estimate("model_devi")[0], 0)
        pool.state["history"]["model_devi"] = [600, 1200, 900]
        self.assertAlmostEqual(pool.estimate("model_devi")[0], 900 * 1.3)
        for bad in (
            {"gpus": 0},
            {"max_allocations": {"train": 0}},
            {"directives": ["--chdir=/x"]},
            {"directives": ["--account={account}"]},
            {"unknown": 1},
            {"time": "soon"},
            {"keep": -1},
            {"renew_before": "48:00:00"},
        ):
            with self.assertRaises(ValueError, msg=bad):
                node_pool.pool_options(dict(POOL, **bad))

    def test_missing_pool_section_is_explained(self):
        (node_pool.pool_root(self.work) / node_pool.CONFIG_FILE).unlink()
        stage, tasks = self.stage("00.train", 1)
        submission = self.submission(stage, tasks, "true")
        with self.assertRaisesRegex(RuntimeError, 'top-level "pool"'):
            submission.run_submission(exit_on_submit=True, clean=False)


if __name__ == "__main__":
    unittest.main()
