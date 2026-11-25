"""FP holdout marking and `dpgen analyze parity`."""

import argparse
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dpdata
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
__package__ = "tests"

from dpgen.generator.lib import holdout  # noqa: E402
from dpgen.tools import analyze, parity  # noqa: E402

from .test_d3_strip import PARAMS, FakeLammps, toy  # noqa: E402

# Stand-in for `dp test`: predictions are the labels plus the offset (eV)
# written in the model file; every call is logged.
FAKE_DP = """
import sys
from pathlib import Path

import dpdata
import numpy as np

args = sys.argv[1:]
model = Path(args[args.index("-m") + 1])
system = dpdata.LabeledSystem(args[args.index("-s") + 1], fmt="deepmd/npy")
prefix = args[args.index("-d") + 1]
offset = float(model.read_text())
energy = system["energies"] / system.get_natoms()
np.savetxt(prefix + ".e_peratom.out", np.c_[energy, energy + offset], header="data_e pred_e")
forces = system["forces"].reshape(-1, 3)
np.savetxt(prefix + ".f.out", np.c_[forces, forces + offset], header="data_f pred_f")
with open(Path(__file__).with_name("calls.log"), "a") as log:
    log.write(f"{model} {args[args.index('-s') + 1]}\\n")
"""


def labeled(nframes, natoms, seed, names=("C", "H")):
    rng = np.random.default_rng(seed)
    cells = np.tile(np.diag([12.0, 12.0, 12.0]), (nframes, 1, 1))
    types = np.arange(natoms) % len(names)
    return dpdata.LabeledSystem(
        data={
            "atom_names": list(names),
            "atom_numbs": [int((types == i).sum()) for i in range(len(names))],
            "atom_types": types,
            "orig": np.zeros(3),
            "cells": cells,
            "coords": rng.random((nframes, natoms, 3)) * 12,
            "energies": rng.normal(size=nframes) - 5.0 * natoms,
            "forces": rng.normal(size=(nframes, natoms, 3)),
            "virials": rng.normal(size=(nframes, 3, 3)),
        }
    )


class TestFpHoldout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def tasks(self, counts):
        paths = []
        for system, count in counts.items():
            for index in range(count):
                path = self.root / f"task.{system:03d}.{index:06d}"
                path.mkdir()
                paths.append(str(path))
        return paths

    def test_marks_a_seeded_share_and_keeps_training_labels(self):
        tasks = self.tasks({0: 200, 1: 1, 2: 2})
        jdata = {"fp_holdout_fraction": 0.2, "fp_style": "vasp"}
        marked = holdout.mark_fp_holdout(tasks, 3, jdata)
        by_system = {s: [t for t in marked if f"task.{s:03d}." in t] for s in (0, 1, 2)}
        self.assertTrue(20 < len(by_system[0]) < 60)
        self.assertEqual(by_system[1], [])  # a single task always trains
        self.assertLess(len(by_system[2]), 2)
        self.assertTrue(all(holdout.is_held_out(t) for t in marked))
        self.assertEqual(sum(holdout.is_held_out(t) for t in tasks), len(marked))
        # Same iteration and seed: same choice.
        for task in marked:
            os.remove(os.path.join(task, holdout.MARKER))
        self.assertEqual(holdout.mark_fp_holdout(tasks, 3, jdata), marked)

    def test_off_plasma_and_invalid_settings(self):
        tasks = self.tasks({0: 10})
        self.assertEqual(holdout.mark_fp_holdout(tasks, 0, {}), [])
        plasma = {
            "fp_holdout_fraction": 0.3,
            "model_devi_jobs": [{"ensemble": "plasma"}],
        }
        self.assertEqual(holdout.mark_fp_holdout(tasks, 0, plasma), [])
        with self.assertRaisesRegex(ValueError, "vasp"):
            holdout.mark_fp_holdout(
                tasks, 0, {"fp_holdout_fraction": 0.3, "fp_style": "gaussian"}
            )
        for bad in (0.6, -0.1, True, float("nan")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                holdout.holdout_options({"fp_holdout_fraction": bad})


class TestParity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = root = Path(self.tmp.name)
        self.jdata = {
            "type_map": ["C", "H"],
            "sys_configs": [["arch/*/*/POSCAR"], ["tol/POSCAR"]],
        }
        (root / "param.json").write_text(json.dumps(self.jdata))
        # Held-out labels from two iterations and two systems.
        self.sets = {
            (0, 0): labeled(3, 6, 1),
            (1, 0): labeled(2, 6, 2),
            (1, 1): labeled(4, 4, 3),
        }
        for (iteration, system), data in self.sets.items():
            path = root / f"iter.{iteration:06d}/02.fp/validation.{system:03d}"
            dpdata.MultiSystems(data, type_map=["C", "H"]).to_deepmd_npy(str(path))
        # Models: iter 0 has offsets 0.004 and 0.002 eV, iter 1 has 0.001.
        for iteration, offsets in ((0, (0.004, 0.002)), (1, (0.001,))):
            train = root / f"iter.{iteration:06d}/00.train"
            train.mkdir(parents=True)
            for index, offset in enumerate(offsets):
                (train / f"graph.{index:03d}.pt2").write_text(str(offset))
        self.fake = root / "fake_dp.py"
        self.fake.write_text(FAKE_DP)
        self.command = f"{sys.executable} {self.fake}"

    def calls(self):
        log = self.root / "calls.log"
        return log.read_text().splitlines() if log.exists() else []

    def analyze(self, **extra):
        args = argparse.Namespace(
            PART="parity",
            JOB_DIR=str(self.root),
            param="param.json",
            output="analysis",
            iters=None,
            frames=False,
            no_plots=False,
            model="0",
            machine=None,
            dp_command=self.command,
            d3_command=None,
            allow_cpu=True,
        )
        vars(args).update(extra)
        return analyze.analyze(args)

    def summary(self):
        with open(self.root / "analysis/parity_summary.csv", newline="") as stream:
            rows = list(csv.DictReader(stream))
        return {(int(r["iter"]), r["system"]): r for r in rows}

    def test_merges_holdouts_and_reports_each_iteration(self):
        notes = self.analyze(iters="0-1")
        rows = self.summary()
        # 9 test frames from 2 iterations; all are tested by both iterations.
        self.assertEqual(int(rows[0, "all"]["frames"]), 9)
        self.assertEqual(int(rows[1, "all"]["frames"]), 9)
        self.assertEqual(int(rows[0, "arch"]["frames"]), 5)
        self.assertEqual(int(rows[0, "tol"]["frames"]), 4)
        self.assertAlmostEqual(float(rows[0, "all"]["energy_rmse_meV_atom"]), 4.0, 3)
        self.assertAlmostEqual(float(rows[1, "all"]["force_rmse_meV_A"]), 1.0, 3)
        self.assertTrue((self.root / "analysis/parity.png").exists())
        self.assertTrue(any("9 held-out frames from 2 iteration" in n for n in notes))
        self.assertTrue(any("4.0 -> 1.0 meV/Å" in n for n in notes))

        # Unchanged test set and models: dp test is not run again.
        before = len(self.calls())
        self.analyze(iters="0-1")
        self.assertEqual(len(self.calls()), before)
        # A new held-out frame changes the test set: everything is retested.
        path = self.root / "iter.000001/02.fp/validation.001"
        dpdata.MultiSystems(labeled(1, 5, 9), type_map=["C", "H"]).to_deepmd_npy(
            str(path)
        )
        self.analyze(iters="1")
        self.assertGreater(len(self.calls()), before)
        self.assertEqual(int(self.summary()[1, "all"]["frames"]), 10)

    def test_committee_mean_and_missing_models(self):
        notes = self.analyze(model="mean")
        rows = self.summary()
        self.assertAlmostEqual(float(rows[0, "all"]["energy_rmse_meV_atom"]), 3.0, 3)
        self.assertEqual(rows[0, "all"]["model"], "committee mean")
        notes = self.analyze(model="1")
        self.assertTrue(any("iter 1 has no trained model 1" in n for n in notes))
        self.assertEqual(sorted({k[0] for k in self.summary()}), [0])

    def test_test_set_is_d3_stripped_when_training_is(self):
        jdata = dict(self.jdata, training_strip_d3=PARAMS)
        out = self.root / "analysis/parity"
        manifest, entries = parity.build_testset(
            self.root, out, jdata, "lmp", FakeLammps()
        )
        self.assertEqual(manifest["d3"], PARAMS)
        merged = dpdata.MultiSystems(type_map=["C", "H"])
        merged.append(self.sets[0, 0], self.sets[1, 0])
        for system, formula, path in entries:
            if system != 0:
                continue
            stripped = dpdata.LabeledSystem(str(path), fmt="deepmd/npy")
            original = merged[formula]
            for index in range(len(original)):
                energy, _, _ = toy(original["coords"][index])
                self.assertAlmostEqual(
                    stripped["energies"][index],
                    original["energies"][index] - energy,
                    5,
                )

    def test_refuses_to_run_without_a_gpu(self):
        with self.assertRaises(SystemExit) as raised:
            with mock.patch.object(
                parity, "check_gpu", return_value=(["PyTorch sees no CUDA GPU"], "")
            ):
                self.analyze(allow_cpu=False)
        self.assertIn("would not run on a GPU", str(raised.exception))
        self.assertIn("--allow-cpu", str(raised.exception))
        self.assertEqual(self.calls(), [])
        with mock.patch.object(
            parity, "check_gpu", return_value=([], "1 CUDA GPU(s) visible")
        ):
            self.analyze(allow_cpu=False)
        self.assertTrue(self.calls())

    def test_gpu_probe(self):
        def probe(count):
            def runner(command, **kwargs):
                self.assertEqual(command[0], sys.executable)
                return mock.Mock(stdout=f"{count}\n")

            return runner

        command = f"{sys.executable} -m deepmd"
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            self.assertEqual(parity.check_gpu(command, probe(1))[0], [])
            problems, _ = parity.check_gpu(command, probe(0))
            self.assertIn("sees no CUDA GPU", problems[0])
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}):
            self.assertIn("hides every GPU", parity.check_gpu(command, probe(1))[0][0])
        with mock.patch.dict(os.environ, {"DEVICE": "cpu"}):
            self.assertTrue(
                any("DEVICE=cpu" in p for p in parity.check_gpu(command, probe(1))[0])
            )
        # A dp console script: the interpreter comes from its shebang.
        script = self.root / "bin" / "dp"
        script.parent.mkdir()
        script.write_text(f"#!{sys.executable}\nprint()\n")
        script.chmod(0o755)
        with mock.patch.dict(os.environ, {"PATH": str(script.parent)}):
            self.assertEqual(parity._dp_python("dp --pt-expt"), sys.executable)

    def test_no_holdout_explains_how_to_get_one(self):
        for path in self.root.glob("iter.*/02.fp"):
            for child in path.iterdir():
                import shutil

                shutil.rmtree(child)
        notes = self.analyze()
        self.assertTrue(any("fp_holdout_fraction" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
