import json
import os
import tempfile
import unittest
from pathlib import Path

import dpdata
import numpy as np

from dpgen.generator.lib import d3
from dpgen.generator.run import revise_lmp_input_pair_coeff

PARAMS = {
    "damping_function": "bj",
    "functional": "pbe",
    "cutoff": 50.0,
    "cn_cutoff": 20.0,
}


def toy(coords):
    """All-pairs harmonic energy: rotation-covariant like D3, easy to check."""
    diff = coords[:, None, :] - coords[None, :, :]
    energy = 0.25 * np.sum(diff**2)
    forces = -diff.sum(axis=1)
    virial = coords.T @ forces
    return energy, forces, virial


def read_data(path):
    lines = Path(path).read_text().splitlines()
    cell = np.zeros((3, 3))
    for line in lines:
        words = line.split()
        if words[-2:] == ["xlo", "xhi"]:
            cell[0, 0] = float(words[1])
        elif words[-2:] == ["ylo", "yhi"]:
            cell[1, 1] = float(words[1])
        elif words[-2:] == ["zlo", "zhi"]:
            cell[2, 2] = float(words[1])
        elif words[-3:] == ["xy", "xz", "yz"]:
            cell[1, 0], cell[2, 0], cell[2, 1] = map(float, words[:3])
    start = lines.index("Atoms # atomic") + 2
    rows = [line.split() for line in lines[start:] if line.strip()]
    return cell, np.array([[float(x) for x in row[2:5]] for row in rows])


class FakeLammps:
    """Stand-in for `lmp -in in.d3`, writing outputs in LAMMPS' own formats."""

    def __init__(self):
        self.calls = 0
        self.cells = []

    def __call__(self, command, workdir):
        self.calls += 1
        script = Path(workdir, "in.d3").read_text()
        assert "pair_style dispersion/d3 bj pbe 50 20" in script, script
        nframes = int(script.split()[3])
        out = []
        for index in range(1, nframes + 1):
            cell, coords = read_data(Path(workdir, f"frame.{index}.data"))
            self.cells.append(cell)
            energy, forces, virial = toy(coords)
            volume = abs(np.linalg.det(cell))
            p = virial * d3.NKTV2P / volume
            out.append(
                f"{index} {energy:.17g} {volume:.17g} {p[0, 0]:.17g} {p[1, 1]:.17g} {p[2, 2]:.17g} "
                f"{p[0, 1]:.17g} {p[0, 2]:.17g} {p[1, 2]:.17g}"
            )
            # Unsorted rows, as a parallel run may write them.
            rows = [
                f"{i + 1} {f[0]:.17g} {f[1]:.17g} {f[2]:.17g}"
                for i, f in enumerate(forces)
            ][::-1]
            Path(workdir, f"forces.{index}").write_text(
                "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n"
                f"{len(rows)}\nITEM: ATOMS id fx fy fz\n" + "\n".join(rows) + "\n"
            )
        Path(workdir, "d3.out").write_text("\n".join(out) + "\n")


def labeled(nframes=3, seed=0):
    rng = np.random.default_rng(seed)
    cells = []
    for _ in range(nframes):
        cell = np.array([[8.0, 0.5, 0.3], [1.0, 7.5, 0.2], [0.4, 1.2, 9.0]])
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        cells.append(cell @ rotation)  # not lower triangular
    cells = np.array(cells)
    fractional = rng.random((nframes, 4, 3))
    return dpdata.LabeledSystem(
        data={
            "atom_names": ["C", "H"],
            "atom_numbs": [2, 2],
            "atom_types": np.array([0, 0, 1, 1]),
            "orig": np.zeros(3),
            "cells": cells,
            "coords": np.einsum("fai,fij->faj", fractional, cells),
            "energies": rng.normal(size=nframes),
            "forces": rng.normal(size=(nframes, 4, 3)),
            "virials": rng.normal(size=(nframes, 3, 3)),
        }
    )


class TestD3Strip(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.original = os.getcwd()
        os.chdir(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(os.chdir, self.original)

    def test_results_rotate_back_to_the_system_frame(self):
        system = labeled()
        fake = FakeLammps()
        energies, forces, virials = d3.compute_d3(system, PARAMS, "lmp", fake)
        for cell in fake.cells:
            self.assertTrue(np.allclose(cell, np.tril(cell)))
        for index in range(len(system)):
            energy, force, virial = toy(system["coords"][index])
            self.assertAlmostEqual(energies[index], energy, places=8)
            np.testing.assert_allclose(forces[index], force, atol=1e-8)
            np.testing.assert_allclose(virials[index], virial, atol=1e-7)
        # The caller's system is not rotated.
        np.testing.assert_array_equal(system["cells"], labeled()["cells"])

    def write_inputs(self, sources):
        inputs = []
        for model in range(2):
            path = Path(f"iter.000000/00.train/{model:03d}/input.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"training": {"training_data": {"systems": sources}}})
            )
            inputs.append(str(path))
        return inputs

    def test_inputs_point_at_stripped_cached_copies(self):
        source = labeled()
        source.to_deepmd_npy("data/sys")
        inputs = self.write_inputs([os.path.abspath("data/sys")])
        jdata = {"training_strip_d3": dict(PARAMS)}
        fake = FakeLammps()
        d3.make_d3_stripped(inputs, "iter.000000/00.train", jdata, {}, fake)
        systems = [
            json.loads(Path(i).read_text())["training"]["training_data"]["systems"]
            for i in inputs
        ]
        self.assertEqual(systems[0], systems[1])
        (path,) = systems[0]
        self.assertTrue(path.startswith("../data.nod3/"))
        stripped = dpdata.LabeledSystem(
            str(Path(inputs[0]).parent / path), fmt="deepmd/npy"
        )
        for index in range(len(source)):
            energy, force, virial = toy(source["coords"][index])
            self.assertAlmostEqual(
                stripped["energies"][index], source["energies"][index] - energy, 6
            )
            np.testing.assert_allclose(
                stripped["forces"][index], source["forces"][index] - force, atol=1e-6
            )
            np.testing.assert_allclose(
                stripped["virials"][index],
                source["virials"][index] - virial,
                atol=1e-6,
            )
        # The original labels are untouched.
        on_disk = dpdata.LabeledSystem("data/sys", fmt="deepmd/npy")
        np.testing.assert_allclose(on_disk["energies"], source["energies"])

        # Same labels next iteration: served from the cache.
        inputs = self.write_inputs([os.path.abspath("data/sys")])
        d3.make_d3_stripped(inputs, "iter.000000/00.train", jdata, {}, fake)
        self.assertEqual(fake.calls, 1)
        # Changed labels: recomputed.
        labeled(seed=1).to_deepmd_npy("data/sys")
        d3.make_d3_stripped(inputs, "iter.000000/00.train", jdata, {}, fake)
        self.assertEqual(fake.calls, 2)

    def test_disabled_leaves_inputs_alone(self):
        inputs = self.write_inputs(["../data.init/sys"])
        before = Path(inputs[0]).read_text()
        for jdata in ({}, {"training_strip_d3": dict(PARAMS, enable=False)}):
            d3.make_d3_stripped(inputs, "iter.000000/00.train", jdata, {}, None)
        self.assertEqual(Path(inputs[0]).read_text(), before)

    def test_options_validate_and_match_exploration(self):
        explore = dict(PARAMS, enable=True, cutoff=50, cn_cutoff=20)
        self.assertEqual(
            d3.strip_options({"training_strip_d3": PARAMS, "lmp_d3": explore}),
            PARAMS,
        )
        with self.assertRaisesRegex(ValueError, "damping_function"):
            d3.strip_options(
                {
                    "training_strip_d3": PARAMS,
                    "lmp_d3": dict(explore, damping_function="original"),
                }
            )
        for key, value in (
            ("damping_function", "zero"),
            ("cutoff", 0),
            ("cn_cutoff", float("nan")),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d3.strip_options({"training_strip_d3": dict(PARAMS, **{key: value})})
        with self.assertRaisesRegex(ValueError, "missing"):
            d3.strip_options({"training_strip_d3": {"functional": "pbe"}})

    def test_non_elements_are_rejected(self):
        system = labeled()
        system.data["atom_names"] = ["Type_0", "Type_1"]
        with self.assertRaisesRegex(ValueError, "type_map.raw"):
            d3.compute_d3(system, PARAMS, "lmp", FakeLammps())


class TestTemplatePairCoeff(unittest.TestCase):
    d3 = {
        "enable": True,
        "damping_function": "bj",
        "functional": "pbe",
        "cutoff": 50.0,
        "cn_cutoff": 20.0,
    }

    def test_pt2_with_d3_names_elements_for_deepmd(self):
        lines = revise_lmp_input_pair_coeff(
            ["pair_style deepmd\n", "pair_coeff * *\n"],
            {"finetune_model_type": "dpa4c", "type_map": ["C", "H"], "lmp_d3": self.d3},
        )
        self.assertIn("pair_coeff      * * deepmd C H\n", lines)
        self.assertIn("pair_coeff      * * dispersion/d3 C H\n", lines)

    def test_non_pt2_with_d3_is_unchanged(self):
        lines = revise_lmp_input_pair_coeff(
            ["pair_style deepmd\n", "pair_coeff * *\n"],
            {"type_map": ["C", "H"], "lmp_d3": self.d3},
        )
        self.assertIn("pair_coeff      * * deepmd\n", lines)


if __name__ == "__main__":
    unittest.main()
