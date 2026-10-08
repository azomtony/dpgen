"""DPA4 foundation fine-tuning through compiled-model exploration."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dpgen.generator.finetune import (
    _get_compress_command,
    _get_train_command,
    _make_train_finetune,
    _run_train_pytorch_with_init,
    prepare_finetune_jdata,
)
from dpgen.generator.lib.lammps import make_lammps_input
from dpgen.generator.lib.model import is_dpa4, uses_pt2
from dpgen.generator.run import (
    _get_model_suffix,
    post_train_dp,
    revise_lmp_input_atom_modify,
    revise_lmp_input_pair_coeff,
)


class TestDPA4Finetune(unittest.TestCase):
    def data(self, checkpoint):
        fixtures = Path(__file__).parent / "generator"
        data = json.loads((fixtures / "param-mg-vasp.json").read_text())
        data.update(
            finetune_model_type="dpa4",
            finetune_model=str(checkpoint),
            init_data_prefix=str((fixtures / "data").resolve()),
        )
        data["default_training_param"]["model"].update(
            type_map=["H", "Mg", "Al"],
            type="dpa4",
            descriptor={"rcut": 6.0},
        )
        return data

    def test_validation_and_backend(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            data = self.data(checkpoint.name)
            prepared = prepare_finetune_jdata(data)
            self.assertTrue(is_dpa4(prepared))
            self.assertTrue(uses_pt2(prepared))
            self.assertEqual(_get_model_suffix(prepared), ".pt2")
            self.assertEqual(
                prepared["training_finetune_model"],
                [checkpoint.name] * data["numb_models"],
            )
            self.assertNotIn("finetune_args", prepared)
            self.assertEqual(_get_train_command(prepared, {}), "dp --pt")
            self.assertEqual(
                _get_train_command(prepared, {"train_command": "dp --pt"}), "dp --pt"
            )
            for command in ("dp --jax", "dp --pt-expt", "dp --pt --pt-expt"):
                with (
                    self.subTest(command=command),
                    self.assertRaisesRegex(RuntimeError, "requires train_command"),
                ):
                    _get_train_command(prepared, {"train_command": command})
            with self.assertRaisesRegex(RuntimeError, "compression"):
                prepare_finetune_jdata(dict(data, dp_compress=True))
            with self.assertRaisesRegex(RuntimeError, "compression"):
                _get_compress_command(prepared, "dp --pt")
            invalid = copy.deepcopy(data)
            invalid["default_training_param"]["model"]["type_map"] = ["H"]
            with self.assertRaisesRegex(RuntimeError, "system elements"):
                prepare_finetune_jdata(invalid)
            invalid["default_training_param"]["model"]["descriptor"] = {}
            with self.assertRaisesRegex(RuntimeError, "complete model"):
                prepare_finetune_jdata(invalid)

    def test_training_input_commands_and_model_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "foundation.pt"
            checkpoint.touch()
            data = prepare_finetune_jdata(self.data(checkpoint))
            model = copy.deepcopy(data["default_training_param"]["model"])
            machine = {
                "deepmd_version": "3.2",
                "api_version": "1.0",
                "train_machine": {},
                "train_resources": {},
            }
            original = os.getcwd()
            os.chdir(tmp)
            try:
                _make_train_finetune(0, data, machine, True)
                seeds = []
                for index in range(data["numb_models"]):
                    task = Path(f"iter.000000/00.train/{index:03d}")
                    generated = json.loads((task / "input.json").read_text())
                    self.assertEqual(generated["model"]["type_map"], model["type_map"])
                    self.assertEqual(generated["model"]["descriptor"]["rcut"], 6.0)
                    seeds.append(generated["model"]["descriptor"]["seed"])
                    self.assertTrue((task / "old/init.pt").is_file())
                    (task / "frozen_model.pt2").touch()
                    (task / "model.ckpt.pt").touch()
                self.assertEqual(len(set(seeds)), data["numb_models"])
                self.assertEqual(data["type_map"], ["Mg", "Al"])
                with patch("dpgen.dispatcher.Dispatcher.make_submission") as submit:
                    _run_train_pytorch_with_init(0, data, machine, True)
                    kwargs = submit.call_args.kwargs
                    self.assertIn("--finetune old/init.pt", kwargs["commands"][0])
                    self.assertIn("--restart model.ckpt", kwargs["commands"][0])
                    self.assertEqual(
                        kwargs["commands"][1],
                        "dp --pt freeze -c model.ckpt.pt -o frozen_model",
                    )
                    self.assertIn("frozen_model.pt2", kwargs["backward_files"])
                    self.assertNotIn("frozen_model.pth", kwargs["backward_files"])
                    _run_train_pytorch_with_init(0, data, machine, False)
                    self.assertIn(
                        "--init-model old/model.ckpt",
                        submit.call_args.kwargs["commands"][0],
                    )
                    self.assertIn(
                        "old/model.ckpt.pt", submit.call_args.kwargs["forward_files"]
                    )
                post_train_dp(0, data, machine)
                self.assertTrue(Path("iter.000000/00.train/graph.000.pt2").is_file())
                data["model_devi_jobs"] = [{}, {}]
                with patch("dpgen.generator.run._check_empty_iter", return_value=False):
                    _make_train_finetune(1, data, machine, False)
                next_task = Path("iter.000001/00.train/000")
                self.assertTrue((next_task / "old/model.ckpt.pt").is_file())
                generated = json.loads((next_task / "input.json").read_text())
                self.assertEqual(generated["model"]["type_map"], model["type_map"])
                with patch("dpgen.generator.run._check_empty_iter", return_value=False):
                    _make_train_finetune(2, data, machine, True)
                self.assertTrue(Path("iter.000002/00.train/000/old/init.pt").is_file())
            finally:
                os.chdir(original)

    def test_exploration_element_and_atom_mapping(self):
        data = {"finetune_model_type": "dpa4", "type_map": ["Mg", "Al"]}
        lines = [
            "atom_style atomic\n",
            "read_data conf.lmp\n",
            "pair_style deepmd graph.000.pt2\n",
            "pair_coeff * *\n",
        ]
        lines = revise_lmp_input_atom_modify(lines, data)
        lines = revise_lmp_input_atom_modify(lines, data)
        result = "".join(revise_lmp_input_pair_coeff(lines, data))
        self.assertEqual(result.count("map yes"), 1)
        self.assertLess(result.index("atom_modify"), result.index("read_data"))
        self.assertIn("pair_coeff      * * Mg Al", result)
        generated = make_lammps_input(
            "nvt",
            "conf.lmp",
            ["graph.000.pt2"],
            100,
            0.001,
            None,
            10,
            [24, 27],
            300,
            jdata=data,
            pres=1.0,
        )
        self.assertIn("pair_coeff      * * Mg Al", generated)
        self.assertIn("map yes", generated)
