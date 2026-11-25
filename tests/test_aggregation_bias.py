"""Colvars aggregation bias for native LAMMPS exploration."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import dpdata
import numpy as np

from dpgen.generator.lib import aggregation
from dpgen.generator.run import make_model_devi

CELL = np.array([[30.0, 0.0, 0.0], [2.0, 30.0, 0.0], [0.0, 0.0, 30.0]])


def coronene(center):
    """C24H12 in the xy plane: a 24-atom fused aromatic core."""
    a1 = np.array([2.46, 0, 0])
    a2 = np.array([1.23, 1.23 * 3**0.5, 0])
    basis = (np.zeros(3), np.array([1.23, 1.23 / 3**0.5, 0]))
    middle = np.array([1.23, -1.23 / 3**0.5, 0]) + a2
    points = np.array(
        [i * a1 + j * a2 + b for i in range(-6, 7) for j in range(-6, 7) for b in basis]
    )
    carbons = points[np.linalg.norm(points - middle, axis=1) < 4.3] - middle
    hydrogens = []
    for atom in carbons:
        near = carbons[np.linalg.norm(carbons - atom, axis=1) < 1.6]
        if len(near) == 3:  # itself and two neighbours: an edge carbon
            outward = atom - near.mean(axis=0)
            hydrogens.append(atom + 1.09 * outward / np.linalg.norm(outward))
    return carbons + center, np.array(hydrogens) + center


def toluene(center):
    angles = np.arange(6) * np.pi / 3
    ring = np.stack([np.cos(angles), np.sin(angles), 0 * angles], axis=1)
    carbons = list(1.39 * ring)
    hydrogens = list(2.48 * ring[1:])
    methyl = 2.90 * ring[0]
    carbons.append(methyl)
    for angle in (0, 2 * np.pi / 3, 4 * np.pi / 3):
        hydrogens.append(
            methyl + np.array([0.36, 1.03 * np.cos(angle), 1.03 * np.sin(angle)])
        )
    return np.array(carbons) + center, np.array(hydrogens) + center


def write_poscar(path, cell=CELL, first=(8, 15, 15), second=(29, 15, 18)):
    """Two coronenes (the second split across the x boundary) and a toluene."""
    parts = [coronene(np.array(first)), coronene(np.array(second))]
    parts.append(toluene(np.array([15.0, 5.0, 5.0])))
    carbons = np.concatenate([c for c, _ in parts])
    hydrogens = np.concatenate([h for _, h in parts])
    coords = np.concatenate([carbons, hydrogens])
    frac = coords @ np.linalg.inv(cell)
    coords = (frac - np.floor(frac)) @ cell  # wrapped, so the second core is split
    system = dpdata.System(
        data={
            "atom_names": ["C", "H"],
            "atom_numbs": [len(carbons), len(hydrogens)],
            "atom_types": np.array([0] * len(carbons) + [1] * len(hydrogens)),
            "orig": np.zeros(3),
            "cells": cell[None],
            "coords": coords[None],
        }
    )
    system.to_vasp_poscar(path)
    return len(carbons), len(hydrogens)


def read_conf_with_images(path):
    lines = Path(path).read_text().splitlines()
    start = lines.index("Atoms # atomic") + 2
    rows = np.array([line.split() for line in lines[start:] if line.strip()], float)
    return rows[:, 2:5], rows[:, 5:8].astype(int)


class TestAggregationBias(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.chdir, self.original)
        self.ncarbon, _ = write_poscar("POSCAR")
        job = {
            "sys_idx": [0],
            "ensemble": "nvt",
            "temps": [300],
            "nsteps": 20,
            "trj_freq": 10,
            "aggregation_bias": {"target": 3.5, "k": 1.0, "pull_steps": 10},
        }
        self.data = {
            "type_map": ["C", "H"],
            "mass_map": [12.011, 1.008],
            "sys_configs": [[str(Path("POSCAR").resolve())]],
            "model_devi_dt": 0.001,
            "model_devi_skip": 0,
            "model_devi_f_trust_lo": 0.1,
            "model_devi_f_trust_hi": 0.3,
            "model_devi_jobs": [copy.deepcopy(job)],
        }
        self.machine = {"deepmd_version": "3.2"}

    def make(self):
        train = Path("iter.000000/00.train")
        train.mkdir(parents=True, exist_ok=True)
        (train / "graph.000.pb").touch()
        self.assertTrue(make_model_devi(0, self.data, self.machine))
        (task,) = sorted(Path("iter.000000/01.model_devi").glob("task.*.*"))
        return task

    def test_detects_two_fused_cores_and_ignores_toluene(self):
        system = dpdata.System("POSCAR", fmt="vasp/poscar", type_map=["C", "H"])
        cores = aggregation.detect_cores(
            system["coords"][0], system["cells"][0], ["C", "H"], system["atom_types"]
        )
        self.assertEqual([len(core) for core in cores], [24, 24])
        self.assertEqual(sorted(cores[0] + cores[1]), list(range(48)))
        # With a 6-atom threshold the toluene ring counts too.
        cores = aggregation.detect_cores(
            system["coords"][0],
            system["cells"][0],
            ["C", "H"],
            system["atom_types"],
            min_core_atoms=6,
        )
        self.assertEqual(sorted(len(core) for core in cores), [6, 24, 24])

    def test_task_gets_colvars_recenter_and_whole_cores(self):
        task = self.make()
        record = json.loads((task / "job.json").read_text())["aggregation_bias"]
        self.assertEqual(record["cores"], [list(range(24)), list(range(24, 48))])
        self.assertEqual(record["core_elements"], [{"C": 24}, {"C": 24}])
        # The second core is 21 Å away along x, i.e. 9 Å via the boundary, 3 Å up.
        self.assertAlmostEqual(record["start_distance"], (9**2 + 3**2) ** 0.5, 5)

        colvars = (task / aggregation.COLVARS_INPUT).read_text()
        self.assertIn("colvarsTrajFrequency 10", colvars)
        self.assertIn("atomNumbers " + " ".join(map(str, range(1, 25))), colvars)
        self.assertIn("atomNumbers " + " ".join(map(str, range(25, 49))), colvars)
        self.assertIn(f"centers {record['start_distance']:.10g}", colvars)
        self.assertIn("targetCenters 3.5", colvars)
        self.assertIn("targetNumSteps 10", colvars)
        self.assertNotIn("harmonicWalls", colvars)

        lammps = (task / "input.lammps").read_text()
        block = lammps.index("group           dpgen_cores id 1:48")
        recenter = lammps.index(
            "fix             dpgen_center dpgen_cores recenter 0.5 0.5 0.5 "
            "units fraction shift all"
        )
        bias = lammps.index("fix             dpgen_bias all colvars colvars.inp")
        self.assertLess(lammps.index("fix             1 all nvt"), block)
        self.assertLess(block, recenter)
        self.assertLess(recenter, bias)
        self.assertLess(bias, lammps.index("\ntimestep"))

        # conf.lmp is now the task's own file, with image flags making each core
        # whole and the second core the image nearest the first.
        conf = task / "conf.lmp"
        self.assertFalse(conf.is_symlink())
        coords, images = read_conf_with_images(conf)
        frac = coords @ np.linalg.inv(CELL)
        self.assertTrue(((frac >= -1e-9) & (frac < 1 + 1e-9)).all())
        unwrapped = (frac + images) @ CELL
        for core in (range(24), range(24, 48)):
            spread = np.ptp(unwrapped[list(core)], axis=0)
            self.assertLess(spread.max(), 10.0)
        distance = np.linalg.norm(
            unwrapped[24:48].mean(axis=0) - unwrapped[:24].mean(axis=0)
        )
        self.assertAlmostEqual(distance, record["start_distance"], 5)
        # Same atoms, same order, same positions modulo the box.
        original = dpdata.System("POSCAR", fmt="vasp/poscar", type_map=["C", "H"])
        delta = (original["coords"][0] - coords) @ np.linalg.inv(CELL)
        np.testing.assert_allclose(delta - np.round(delta), 0, atol=1e-6)

    def test_explicit_cores_wall_and_no_recenter(self):
        self.data["model_devi_jobs"][0]["aggregation_bias"] = {
            "cores": {"0": [list(range(24, 48)), list(range(24))]},
            "upper_wall": 12.0,
            "recenter": False,
        }
        task = self.make()
        record = json.loads((task / "job.json").read_text())["aggregation_bias"]
        self.assertEqual(record["cores"], [list(range(24, 48)), list(range(24))])
        self.assertEqual(record["pull_steps"], 10)  # default: half of nsteps
        colvars = (task / aggregation.COLVARS_INPUT).read_text()
        self.assertIn("upperWalls 12", colvars)
        self.assertNotIn("recenter", (task / "input.lammps").read_text())

    def test_jobs_without_the_bias_are_unchanged(self):
        del self.data["model_devi_jobs"][0]["aggregation_bias"]
        task = self.make()
        self.assertTrue((task / "conf.lmp").is_symlink())
        self.assertFalse((task / aggregation.COLVARS_INPUT).exists())
        self.assertNotIn("colvars", (task / "input.lammps").read_text())
        self.assertNotIn(
            "aggregation_bias", json.loads((task / "job.json").read_text())
        )

    def test_rejects_starts_near_half_the_box(self):
        small = np.diag([16.0, 30.0, 30.0])
        write_poscar("POSCAR", cell=small, first=(4, 15, 10), second=(12, 15, 17))
        with self.assertRaisesRegex(ValueError, "half the box"):
            self.make()

    def test_wrong_number_of_cores_is_explained(self):
        self.data["model_devi_jobs"][0]["aggregation_bias"]["min_core_atoms"] = 6
        with self.assertRaisesRegex(ValueError, "found 3 cores"):
            self.make()

    def test_option_validation(self):
        base = {"aggregation_bias": {}}
        self.assertEqual(aggregation.options(base, 100)["pull_steps"], 50)
        self.assertIsNone(aggregation.options({}, 100))
        for bad in (
            {"target": 0},
            {"k": -1},
            {"pull_steps": 200},
            {"pull_steps": 1.5},
            {"upper_wall": 3.0},
            {"recenter": "yes"},
            {"min_core_atoms": 2},
            {"cores": "first"},
            {"unknown": 1},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                aggregation.options({"aggregation_bias": bad}, 100)
        job = {"aggregation_bias": {"cores": [[0], [1]]}}
        for jdata, extra in (
            ({"model_devi_engine": "gromacs"}, {}),
            ({}, {"template": {}}),
            ({}, {"nbeads": 4}),
            ({"model_devi_nopbc": True}, {}),
            ({"shuffle_poscar": True}, {}),
        ):
            with self.subTest(jdata=jdata, extra=extra), self.assertRaises(ValueError):
                aggregation.check_compatible(jdata, dict(job, **extra))

    def test_bad_explicit_indices(self):
        for cores in ([[0, 1]], [[0, 1], [1, 2]], [[0], [999]], [[], [1]]):
            self.data["model_devi_jobs"][0]["aggregation_bias"]["cores"] = cores
            with self.subTest(cores=cores), self.assertRaises(ValueError):
                self.make()


if __name__ == "__main__":
    unittest.main()
