"""Condition-matched continuation of native LAMMPS exploration."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dpdata
import numpy as np

from dpgen.generator.lib.exploration_restart import trajectory_cleanup_index
from dpgen.generator.run import make_model_devi, post_fp


def dump(step, x):
    return (
        f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        f"ITEM: ATOMS id type x y z\n1 1 {x} 1 1\n"
    )


class TestExplorationRestart(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.chdir, self.original)
        Path("initial.dump").write_text(dump(0, 0.5))
        initial = dpdata.System("initial.dump", fmt="lammps/dump", type_map=["H"])
        initial.to_vasp_poscar("POSCAR")
        initial.to_vasp_poscar("POSCAR-other")
        job = {
            "sys_idx": [0],
            "ensemble": "nvt",
            "temps": [300, 600],
            "nsteps": 20,
            "trj_freq": 10,
        }
        self.data = {
            "type_map": ["H"],
            "mass_map": [1.0],
            "sys_configs": [[str(Path("POSCAR").resolve())]],
            "model_devi_dt": 0.001,
            "model_devi_skip": 0,
            "model_devi_f_trust_lo": 0.1,
            "model_devi_f_trust_hi": 0.3,
            "model_devi_jobs": [copy.deepcopy(job) for _ in range(3)],
        }
        self.machine = {"deepmd_version": "3.2"}

    def make(self, iteration):
        train = Path(f"iter.{iteration:06d}/00.train")
        train.mkdir(parents=True, exist_ok=True)
        (train / "graph.000.pb").touch()
        self.assertTrue(make_model_devi(iteration, self.data, self.machine))
        return sorted(Path(f"iter.{iteration:06d}/01.model_devi").glob("task.*.*"))

    def trajectory(self, task, values=None, merged=False):
        if values is None:
            values = [(0, 0.05, 1), (10, 0.2, 2), (20, 0.9, 3)]
        rows = [[step, 0, 0, 0, force, 0, 0] for step, force, _ in values]
        np.savetxt(task / "model_devi.out", rows)
        if merged:
            (task / "all.lammpstrj").write_text(
                "".join(dump(s, x) for s, _, x in values)
            )
        else:
            for step, _, x in values:
                (task / "traj" / f"{step}.lammpstrj").write_text(dump(step, x))

    def coordinate(self, task):
        system = dpdata.System(str(task / "conf.lmp"), fmt="lammps/lmp", type_map=["H"])
        return system.data["coords"][0, 0, 0]

    def test_backward_search_matches_temperature_and_legacy_source(self):
        previous = self.make(0)  # Legacy job.json has no source_conf.
        self.trajectory(previous[0])
        self.trajectory(previous[1], [(0, 0.05, 4), (10, 0.1, 5), (20, 0.2, 6)])
        self.data["model_devi_restart_from_last"] = True
        self.data["model_devi_jobs"][1]["temps"] = [600, 300, 900]
        tasks = self.make(1)
        self.assertAlmostEqual(self.coordinate(tasks[0]), 6)
        self.assertAlmostEqual(self.coordinate(tasks[1]), 2)
        self.assertAlmostEqual(self.coordinate(tasks[2]), 0.5)
        self.assertFalse((tasks[1] / "conf.lmp").is_symlink())
        self.assertTrue((tasks[2] / "conf.lmp").is_symlink())
        provenance = json.loads((tasks[1] / "job.json").read_text())
        self.assertEqual(provenance["restart"]["step"], 10)
        self.assertEqual(provenance["source_conf"], str(Path("POSCAR").resolve()))
        self.assertIn("velocity", (tasks[1] / "input.lammps").read_text())
        # Continue again from tasks whose conf.lmp is no longer a symlink.
        self.trajectory(tasks[1], [(0, 0.05, 7)])
        next_tasks = self.make(2)
        self.assertAlmostEqual(self.coordinate(next_tasks[0]), 7)

    def test_disabled_keeps_initial_structure(self):
        for task in self.make(0):
            self.trajectory(task)
        for task in self.make(1):
            self.assertTrue((task / "conf.lmp").is_symlink())
            self.assertAlmostEqual(self.coordinate(task), 0.5)
            self.assertNotIn("restart", json.loads((task / "job.json").read_text()))

    def test_merged_trajectory_uses_actual_timesteps(self):
        self.data["model_devi_merge_traj"] = True
        previous = self.make(0)
        self.trajectory(
            previous[0], [(0, 0.05, 1), (7, 0.2, 2), (19, 0.8, 3)], merged=True
        )
        self.data["model_devi_restart_from_last"] = True
        tasks = self.make(1)
        self.assertAlmostEqual(self.coordinate(tasks[0]), 2)
        self.assertAlmostEqual(self.coordinate(tasks[1]), 0.5)

    def test_unreadable_candidate_and_missing_dump_search_backward(self):
        previous = self.make(0)
        self.trajectory(previous[0], [(0, 0.05, 1), (10, 0.2, 2), (20, 0.2, 3)])
        (previous[0] / "traj/20.lammpstrj").write_text("truncated")
        (previous[0] / "traj/10.lammpstrj").unlink()
        self.data["model_devi_restart_from_last"] = True
        self.assertAlmostEqual(self.coordinate(self.make(1)[0]), 1)

    def test_truncated_merged_last_frame_searches_backward(self):
        self.data["model_devi_merge_traj"] = True
        previous = self.make(0)
        self.trajectory(previous[0], [(0, 0.05, 1), (10, 0.2, 2)], merged=True)
        with (previous[0] / "all.lammpstrj").open("a") as stream:
            stream.write("ITEM: TIMESTEP\n20\nITEM: NUMBER OF ATOMS\n")
        with (previous[0] / "model_devi.out").open("a") as stream:
            stream.write("20 0 0 0 0.2 0 0\n")
        self.data["model_devi_restart_from_last"] = True
        self.assertAlmostEqual(self.coordinate(self.make(1)[0]), 2)

    def test_electronic_temperature_must_match(self):
        self.data["use_ele_temp"] = 1
        self.data["model_devi_jobs"][0]["temps"] = [[300, 1000]]
        previous = self.make(0)
        self.trajectory(previous[0])
        self.data["model_devi_jobs"][1]["temps"] = [[300, 2000], [300, 1000]]
        self.data["model_devi_restart_from_last"] = True
        tasks = self.make(1)
        self.assertAlmostEqual(self.coordinate(tasks[0]), 0.5)
        self.assertAlmostEqual(self.coordinate(tasks[1]), 2)

    def test_normalized_force_deviation_uses_labeling_values(self):
        previous = self.make(0)
        self.trajectory(previous[0])
        rows = np.loadtxt(previous[0] / "model_devi.out")
        rows[:, 4:7] /= 10
        np.savetxt(previous[0] / "model_devi_avgf.out", rows)
        self.data["model_devi_f_avg_relative"] = True
        self.data["model_devi_restart_from_last"] = True
        self.assertAlmostEqual(self.coordinate(self.make(1)[0]), 3)

    def test_previous_job_system_force_override_and_skip(self):
        previous = self.make(0)
        self.trajectory(previous[0], [(0, 0.05, 1), (10, 0.2, 2), (20, 0.25, 3)])
        self.data["model_devi_f_trust_hi"] = [0.15]
        self.data["model_devi_jobs"][0]["model_devi_f_trust_hi"] = {"0": 0.25}
        self.data["model_devi_skip"] = 5
        self.data["model_devi_restart_from_last"] = True
        self.assertAlmostEqual(self.coordinate(self.make(1)[0]), 2)

    def test_upper_limits_and_nonfinite_values_fall_back(self):
        previous = self.make(0)
        self.trajectory(previous[0], [(0, 0.3, 1), (10, float("nan"), 2), (20, 0.2, 3)])
        rows = np.loadtxt(previous[0] / "model_devi.out")
        rows[-1, 1] = 0.5  # Candidate force, failed virial: reject the frame.
        np.savetxt(previous[0] / "model_devi.out", rows)
        self.data["model_devi_jobs"][0]["model_devi_v_trust_hi"] = {"0": 0.5}
        self.data["model_devi_restart_from_last"] = True
        self.assertAlmostEqual(self.coordinate(self.make(1)[0]), 0.5)

    def test_pressure_ensemble_source_and_system_must_match(self):
        for mismatch in ("pressure", "ensemble", "source", "system"):
            with self.subTest(mismatch=mismatch):
                # Isolate each matching check in its own working directory.
                with tempfile.TemporaryDirectory(dir=".") as folder:
                    original = os.getcwd()
                    os.chdir(folder)
                    try:
                        data = copy.deepcopy(self.data)
                        self.data["model_devi_jobs"][0].update(
                            ensemble="npt", press=[1]
                        )
                        for task in self.make(0):
                            self.trajectory(task)
                        self.data["model_devi_jobs"][1].update(
                            ensemble="npt", press=[1]
                        )
                        if mismatch == "pressure":
                            self.data["model_devi_jobs"][1]["press"] = [2]
                        elif mismatch == "ensemble":
                            self.data["model_devi_jobs"][1]["ensemble"] = "nvt"
                        elif mismatch == "source":
                            self.data["sys_configs"][0] = [
                                str(Path(original) / "POSCAR-other")
                            ]
                        else:
                            self.data["sys_configs"].append(self.data["sys_configs"][0])
                            self.data["model_devi_jobs"][1]["sys_idx"] = [1]
                        self.data["model_devi_restart_from_last"] = True
                        self.assertAlmostEqual(self.coordinate(self.make(1)[0]), 0.5)
                    finally:
                        self.data = data
                        os.chdir(original)

    def test_cleanup_retains_latest_trajectory(self):
        tasks0 = self.make(0)
        tasks1 = self.make(1)
        self.data.update(fp_style="custom", model_devi_restart_from_last=True)
        with (
            patch("dpgen.generator.run.post_fp_custom"),
            patch("dpgen.generator.run.post_fp_check_fail"),
        ):
            post_fp(1, self.data)
        self.assertFalse((tasks0[0] / "traj").exists())
        self.assertTrue((tasks1[0] / "traj").is_dir())
        self.assertIsNone(trajectory_cleanup_index(0, True, True))
        self.assertIsNone(trajectory_cleanup_index(2, False, True))
        self.assertEqual(trajectory_cleanup_index(2, 0, True), 1)
        self.assertEqual(trajectory_cleanup_index(3, 2, True), 1)
        self.assertEqual(trajectory_cleanup_index(2, True, False), 2)

    def test_unsupported_modes_are_explicit(self):
        self.data["model_devi_restart_from_last"] = True
        self.data["model_devi_jobs"][0]["nbeads"] = 4
        with self.assertRaisesRegex(ValueError, "native classical LAMMPS"):
            self.make(0)
