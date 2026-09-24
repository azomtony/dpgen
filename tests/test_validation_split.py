import json
import os
import tempfile
import unittest
from pathlib import Path

import dpdata
import numpy as np

from dpgen.generator.lib.validation import make_validation_split, validation_options
from dpgen.util import convert_training_data_to_hdf5


class TestValidationSplit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        previous = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, previous)
        self.root = Path.cwd()
        self.system = dpdata.LabeledSystem(
            data={
                "atom_names": ["Cu"],
                "atom_numbs": [1],
                "atom_types": np.array([0]),
                "orig": np.zeros(3),
                "cells": np.tile(np.eye(3) * 10, (20, 1, 1)),
                "coords": np.zeros((20, 1, 3)),
                "energies": np.arange(20, dtype=float),
                "forces": np.zeros((20, 1, 3)),
            }
        )
        self.system.to_deepmd_npy("source")
        self.options = {"validation_fraction": 0.1, "validation_seed": 42}

    def inputs(self, iteration):
        work = Path(f"iter.{iteration:06d}/00.train")
        files = []
        for model in range(2):
            path = work / f"{model:03d}" / "input.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "training": {
                            "seed": model,
                            "training_data": {
                                "systems": [str(self.root / "source")],
                                "batch_size": [20],
                            },
                        }
                    }
                )
            )
            files.append(str(path))
        return work, files

    def test_disjoint_persistent_shared_split_and_hdf5(self):
        for iteration in (0, 1):
            work, files = self.inputs(iteration)
            make_validation_split(files, work, self.options)
            configs = [json.loads(Path(file).read_text()) for file in files]
            self.assertEqual(
                configs[0]["training"]["training_data"],
                configs[1]["training"]["training_data"],
            )
            self.assertEqual(
                configs[0]["training"]["validation_data"],
                configs[1]["training"]["validation_data"],
            )
            training = configs[0]["training"]
            directory = Path(files[0]).parent
            train = dpdata.LabeledSystem(
                str(directory / training["training_data"]["systems"][0]),
                fmt="deepmd/npy",
            )
            val = dpdata.LabeledSystem(
                str(directory / training["validation_data"]["systems"][0]),
                fmt="deepmd/npy",
            )
            self.assertEqual(len(train), 18)
            self.assertEqual(len(val), 2)
            self.assertFalse(set(train["energies"]) & set(val["energies"]))
            self.assertEqual(
                set(train["energies"]) | set(val["energies"]), set(range(20))
            )
            self.assertEqual(training["training_data"]["batch_size"], [18])
            if iteration == 0:
                held_out = val["energies"].copy()
            else:
                np.testing.assert_array_equal(held_out, val["energies"])
            convert_training_data_to_hdf5(files, str(work / "data.hdf5"))
            converted = json.loads(Path(files[0]).read_text())["training"]
            h5path, group = converted["training_data"]["systems"][0].split("#")
            h5train = dpdata.LabeledSystem(
                str(directory / h5path) + "#" + group, fmt="deepmd/hdf5"
            )
            self.assertFalse(set(h5train["energies"]) & set(held_out))
        self.assertEqual(len(list(Path(".dpgen-validation").glob("*/split.json"))), 1)

    def test_reject_changed_split_and_options(self):
        work, files = self.inputs(0)
        make_validation_split(files, work, self.options)
        work, files = self.inputs(1)
        with self.assertRaisesRegex(ValueError, "changed"):
            make_validation_split(files, work, dict(self.options, validation_seed=99))
        for fraction in (-0.1, 1, float("nan"), True):
            with self.assertRaises(ValueError):
                validation_options({"validation_fraction": fraction})

    def test_make_train_integration_and_batch_transfer(self):
        from unittest.mock import patch

        from dpgen.generator.run import make_train, run_train_dp

        fixture = Path(__file__).resolve().parent / "generator"
        data = json.loads((fixture / "param-mg-vasp.json").read_text())
        data.update(self.options)
        data["init_data_prefix"] = str(fixture / "data")
        machine = {
            "deepmd_version": "3.2",
            "api_version": "1.0",
            "train_machine": {},
            "train_resources": {},
        }
        make_train(0, data, machine)
        inputs = sorted(Path("iter.000000/00.train").glob("[0-9]*/input.json"))
        self.assertEqual(len(inputs), 4)
        for file in inputs:
            config = json.loads(file.read_text())["training"]
            self.assertTrue(config["validation_data"]["systems"])
            self.assertTrue(
                all("data.validation" in s for s in config["training_data"]["systems"])
            )
        with patch("dpgen.generator.run.make_submission") as submit:
            run_train_dp(0, data, machine)
            self.assertIn(
                "data.validation", submit.call_args.kwargs["forward_common_files"]
            )

    def test_disabled_unchanged(self):
        work, files = self.inputs(0)
        original = Path(files[0]).read_bytes()
        make_validation_split(files, work, {})
        self.assertEqual(original, Path(files[0]).read_bytes())
        self.assertFalse(Path(".dpgen-validation").exists())
