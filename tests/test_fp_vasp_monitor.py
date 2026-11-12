"""SCF watchdog: real process shutdown, trend decisions and label exclusion."""

import json
import os
import shlex
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dpgen.dispatcher import fp_recovery_worker as worker
from dpgen.dispatcher.fp_vasp_monitor import Monitor, options, scf_record


def row(step, de, deps=None):
    return f"RMM: {step} -0.173874679532E+04 {de} {de if deps is None else deps} 12172 0.614E-02 0.387E+01\n"


class TestVaspMonitor(unittest.TestCase):
    def setUp(self):
        previous = os.getcwd()
        temporary = tempfile.TemporaryDirectory()
        os.chdir(temporary.name)
        self.addCleanup(temporary.cleanup)
        self.addCleanup(os.chdir, previous)
        Path("INCAR").write_text("NSW=0; NELM=300; EDIFF=1E-6\n")
        self.now = 0.0

    def monitor(self, **settings):
        return Monitor(dict(enabled=True, **settings), clock=lambda: self.now)

    def test_vasp_columns_and_fortran_numbers(self):
        record = scf_record(row(84, "0.79795E+01", "-0.20400E-01"))
        self.assertEqual(record["step"], 84)
        self.assertAlmostEqual(record["de"], 7.9795)
        self.assertAlmostEqual(record["charge_rms"], 3.87)
        self.assertEqual(scf_record("CG : 3 -1D+03 1D-02 -2D-03 40 1E-3\n")["de"], 0.01)
        self.assertIsNone(scf_record("ordinary VASP header"))

    def test_healthy_improvement_and_oscillation_near_convergence_continue(self):
        monitor = self.monitor(stagnation_window=5, min_scf_steps=10)
        for step in range(1, 70):
            monitor.feed(row(step, (-1) ** step * 0.8**step))
        self.assertIsNone(monitor.reason)
        for step in range(70, 100):
            monitor.feed(row(step, 1e-7, -1e-7))
        self.assertIsNone(monitor.reason)
        self.assertFalse(Path("STOPCAR").exists())

    def test_plateau_requires_history_and_being_far_from_ediff(self):
        monitor = self.monitor(stagnation_window=5, min_scf_steps=20)
        for step in range(1, 20):
            monitor.feed(row(step, 0.5 * (-1) ** step))
        self.assertIsNone(monitor.reason)
        monitor.feed(row(20, -0.5))
        self.assertIn("stagnation", monitor.reason)
        self.assertEqual(Path("STOPCAR").read_text(), "LABORT = .TRUE.\n")
        self.assertEqual(
            json.loads(Path("fp-monitor-stop.json").read_text())["last_scf"]["step"], 20
        )

    def test_supplied_unstable_rmm_sequence_is_detected(self):
        monitor = self.monitor(
            stagnation_window=20, min_scf_steps=60, max_scf_steps=200
        )
        errors = [
            7.9795,
            3.0146,
            -4.8793,
            7.2762,
            -1.9270,
            -0.15052,
            0.78346,
            -5.4883,
            5.3579,
            -0.049144,
            0.98446,
            0.47866,
            6.1014,
            1.4024,
            -0.72716,
            2.2173,
            -1.4670,
            -6.6964,
            3.9083,
            0.41387,
            1.8717,
            0.94831,
            1.5434,
            -0.27737,
            0.65017,
            0.24548,
            0.13981,
            -290.36,
            -70.786,
            -39.629,
            -42.085,
            -102.21,
            -42.309,
            -1.7897,
            4.9422,
            -87.189,
            -239.95,
            468.92,
            76.926,
            15.197,
            115.57,
            106.32,
            110.64,
        ]
        for step, error in enumerate(errors, 84):
            monitor.feed(row(step, error))
        self.assertIn("stagnation", monitor.reason)

    def test_iteration_cap_and_last_iteration_convergence(self):
        monitor = self.monitor(max_scf_steps=3)
        for step in range(1, 4):
            monitor.feed(row(step, 1))
        self.assertIn("iteration limit 3", monitor.reason)
        Path("STOPCAR").unlink()
        monitor = self.monitor(max_scf_steps=3)
        for step in range(1, 4):
            monitor.feed(row(step, 1e-8 if step == 3 else 1))
        self.assertIsNone(monitor.reason)

    def test_nonfinite_residual_stops_with_valid_json(self):
        monitor = self.monitor()
        monitor.feed(row(1, "NaN"))
        self.assertIn("Nonfinite", monitor.reason)
        self.assertIsNone(
            json.loads(Path("fp-monitor-stop.json").read_text())["last_scf"]["de"]
        )

    def test_startup_timeout_escalates_only_own_child_group(self):
        monitor = self.monitor(
            startup_timeout=5, stop_grace_period=2, kill_grace_period=1
        )
        monitor.process = SimpleNamespace(pid=7654321, poll=lambda: None)
        self.now = 5
        monitor.tick()
        self.assertIn("startup timeout", monitor.reason)
        with patch("dpgen.dispatcher.fp_vasp_monitor.os.killpg") as kill:
            self.now = 7
            monitor.tick()
            kill.assert_called_once_with(7654321, signal.SIGTERM)
            self.now = 8
            monitor.tick()
            self.assertEqual(kill.call_args.args, (7654321, signal.SIGKILL))

    def test_output_growth_defers_stall_timeout(self):
        monitor = self.monitor(startup_timeout=0, stall_timeout=10)
        monitor.feed(row(1, 1))
        self.now = 8
        Path("OUTCAR").write_text("progress")
        monitor.tick()
        self.now = 12
        monitor.tick()
        self.assertIsNone(monitor.reason)
        self.now = 19
        monitor.tick()
        self.assertIn("stall timeout", monitor.reason)

    def test_old_oszicar_is_not_replayed_and_new_partial_lines_are_safe(self):
        Path("OSZICAR").write_text(row(300, 1))
        monitor = self.monitor(max_scf_steps=3)
        monitor.feed(row(1, 1))
        monitor.tick()
        self.assertIsNone(monitor.reason)
        Path("OSZICAR").write_text(row(2, 1).rstrip())
        self.now = 1
        monitor.tick()
        self.assertEqual(monitor.last_step, 1)
        with Path("OSZICAR").open("a") as stream:
            stream.write("\n")
        self.now = 2
        monitor.tick()
        self.assertEqual(monitor.last_step, 2)

    def run_worker(self, script, **policy):
        worker.atomic_json(
            "fp-recovery.json",
            {
                "command": f"{shlex.quote(sys.executable)} -u -c {shlex.quote(script)}",
                "bad_gpu_uuids": [],
                "fp_monitor": dict(
                    enabled=True,
                    check_interval=0.01,
                    stop_grace_period=0.1,
                    kill_grace_period=0.1,
                    **policy,
                ),
            },
        )
        with (
            patch.object(worker, "gpu_inventory", return_value=([], "")),
            patch.object(worker, "cuda_visible_uuids", return_value=(None, "")),
        ):
            code = worker.run_attempt()
        return code, json.loads(Path("fp-attempt.json").read_text())

    def test_real_child_obeys_stopcar_and_no_unconverged_success_is_reported(self):
        script = (
            "import pathlib,time\n"
            + "print('"
            + row(1, 1).strip()
            + "',flush=True)\n"
            + "while not pathlib.Path('STOPCAR').exists(): time.sleep(0.01)\n"
        )
        code, report = self.run_worker(script, max_scf_steps=1)
        self.assertNotEqual(code, 0)
        self.assertEqual(report["category"], "scientific_stop")
        self.assertTrue(Path("fp-monitor-stop.json").exists())
        self.assertNotIn("faulty_gpu_uuids", report)

    def test_real_silent_child_is_terminated_within_watchdog_budget(self):
        started = time.monotonic()
        code, report = self.run_worker(
            "import time; time.sleep(30)", startup_timeout=0.05
        )
        self.assertLess(time.monotonic() - started, 5)
        self.assertNotEqual(code, 0)
        self.assertEqual(report["category"], "scientific_stop")

    def test_real_improving_child_exits_normally(self):
        code, report = self.run_worker(
            "print('" + row(1, 1).strip() + "'); print('" + row(2, 1e-8).strip() + "')",
            max_scf_steps=2,
        )
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "exited")
        self.assertFalse(Path("fp-monitor-stop.json").exists())

    def test_monitor_stopped_outputs_are_excluded_even_if_parseable(self):
        from dpgen.generator.run import _vasp_check_fin, post_fp_vasp

        task = Path("iter.000000/02.fp/task.000.000000")
        task.mkdir(parents=True)
        (task / "OUTCAR").write_text("Elapse\n")
        (task / "fp-monitor-stop.json").write_text("{}")
        self.assertFalse(_vasp_check_fin(str(task)))
        data = {"type_map": ["Cu"], "model_devi_jobs": [{}], "failed_ratio": 1.0}
        with patch("dpgen.generator.run.dpdata.LabeledSystem") as parse:
            post_fp_vasp(0, data)
            parse.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "Too many FP tasks"):
            post_fp_vasp(0, dict(data, failed_ratio=0.1))

    def test_policy_validation_and_single_point_scope(self):
        for setting in (
            {"enabled": "true"},
            {"check_interval": 0},
            {"stagnation_window": 1.5},
            {"min_relative_improvement": 1},
        ):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                options(setting)
        Path("INCAR").write_text("NSW=3")
        with self.assertRaisesRegex(ValueError, "single-point"):
            self.monitor()
        Path("INCAR").write_text("NSW=1; EDIFF=0")
        with self.assertRaisesRegex(ValueError, "EDIFF"):
            self.monitor()

    def test_discard_ledger_excludes_partial_download_without_stop_marker(self):
        from dpgen.dispatcher.fp_vasp_monitor import stopped_task

        task = Path("02.fp/task.000.000001")
        task.mkdir(parents=True)
        (task.parent / ".fp-recovery.json").write_text(
            json.dumps({"tasks": {task.name: {"status": "discarded"}}})
        )
        self.assertTrue(stopped_task(task))

    def test_standalone_compute_node_worker_imports_copied_monitor(self):
        import shutil
        import subprocess

        from dpgen.dispatcher import fp_vasp_monitor

        shutil.copyfile(worker.__file__, "fp_recovery_worker.py")
        shutil.copyfile(fp_vasp_monitor.__file__, "fp_vasp_monitor.py")
        script = "print('" + row(1, 1).strip() + "')"
        worker.atomic_json(
            "fp-recovery.json",
            {
                "command": f"{shlex.quote(sys.executable)} -u -c {shlex.quote(script)}",
                "bad_gpu_uuids": [],
                "fp_monitor": {"enabled": True, "max_scf_steps": 1},
            },
        )
        result = subprocess.run(
            [sys.executable, "fp_recovery_worker.py"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 86, result.stderr)
        report = json.loads(Path("fp-attempt.json").read_text())
        self.assertEqual(report["category"], "scientific_stop")
