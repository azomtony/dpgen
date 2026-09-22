"""DPA4C regular training command and exploration integration tests."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dpgen.generator.lib.model import is_dpa4c, prepare_training_backend
from dpgen.generator.run import (
    _get_model_suffix,
    make_train,
    post_train_dp,
    revise_lmp_input_atom_modify,
    revise_lmp_input_pair_coeff,
    run_train_dp,
)


class TestDPA4CTraining(unittest.TestCase):
    def data(self):
        return {
            "numb_models": 1,
            "type_map": ["Cu"],
            "default_training_param": {"model": {"descriptor": {"type": "dpa4c"}}},
            "one_h5": True,
            "dp_compress": True,
        }

    def test_detection_and_backend(self):
        data = self.data()
        self.assertTrue(is_dpa4c(data))
        prepare_training_backend(data)
        self.assertEqual(data["train_backend"], "pytorch")
        self.assertEqual(_get_model_suffix(data), ".pt2")
        self.assertEqual(data["type_map"], ["Cu"])
        self.assertNotIn("training_finetune_model", data)
        data["train_backend"] = "tensorflow"
        with self.assertRaises(ValueError):
            prepare_training_backend(data)
        self.assertFalse(is_dpa4c({}))
        self.assertEqual(_get_model_suffix({"train_backend": "pytorch"}), ".pth")

    def test_reject_conflicting_backend_command(self):
        for command in ("dp --pt", "dp --jax", "dp --pt-expt --pt"):
            with self.subTest(command=command), self.assertRaises(ValueError):
                run_train_dp(
                    0, self.data(), {"deepmd_version": "3.2", "train_command": command}
                )

    def test_train_restart_freeze_compress_and_reuse(self):
        data = self.data()
        machine = {
            "deepmd_version": "3.2",
            "api_version": "1.0",
            "train_command": "dp --pt-expt",
            "train_machine": {},
            "train_resources": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            original = os.getcwd()
            os.chdir(tmp)
            try:
                task = Path("iter.000000/00.train/000")
                task.mkdir(parents=True)
                with patch("dpgen.generator.run.make_submission") as submit:
                    run_train_dp(0, data, machine)
                    kwargs = submit.call_args.kwargs
                    commands = kwargs["commands"]
                    self.assertIn(
                        "dp --pt-expt train input.json --skip-neighbor-stat",
                        commands[0],
                    )
                    self.assertIn("model.ckpt.pt", commands[0])
                    self.assertIn("--restart model.ckpt", commands[0])
                    self.assertNotIn("--finetune", commands[0])
                    self.assertNotIn("--use-pretrain-script", commands[0])
                    self.assertNotIn("--pt ", commands[0])
                    self.assertEqual(
                        commands[1],
                        "dp --pt-expt freeze -c model.ckpt.pt -o frozen_model --lower-kind graph",
                    )
                    self.assertIn("compressed_model.pt2", commands[2])
                    self.assertIn("compressed_model.pt2", kwargs["backward_files"])
                    self.assertIn("model.ckpt.pt", kwargs["backward_files"])
                    submit.return_value.run_submission.assert_called_once()
                    (task / "compressed_model.pt2").touch()
                    post_train_dp(0, data, machine)
                    self.assertTrue(
                        Path("iter.000000/00.train/graph.000.pt2").is_file()
                    )
                    data["training_init_model"] = True
                    run_train_dp(0, data, machine)
                    self.assertIn(
                        "--init-model old/model.ckpt",
                        submit.call_args.kwargs["commands"][0],
                    )
                    self.assertIn(
                        "old/model.ckpt.pt", submit.call_args.kwargs["forward_files"]
                    )
            finally:
                os.chdir(original)

    def test_make_train_preserves_scratch_model_and_system_type_map(self):
        fixtures = Path(__file__).parent / "generator"
        data = json.loads((fixtures / "param-mg-vasp.json").read_text())
        data["init_data_prefix"] = str((fixtures / "data").resolve())
        data["default_training_param"]["model"]["descriptor"] = {"type": "dpa4c"}
        prepare_training_backend(data)
        with tempfile.TemporaryDirectory() as tmp:
            original = os.getcwd()
            os.chdir(tmp)
            try:
                make_train(0, data, {"deepmd_version": "3.2"})
                models = []
                for index in range(data["numb_models"]):
                    generated = json.loads(
                        Path(f"iter.000000/00.train/{index:03d}/input.json").read_text()
                    )
                    self.assertEqual(generated["model"]["type_map"], ["Mg", "Al"])
                    self.assertEqual(generated["model"]["descriptor"]["type"], "dpa4c")
                    models.append(generated["model"]["descriptor"]["seed"])
                self.assertEqual(len(set(models)), data["numb_models"])
            finally:
                os.chdir(original)

    def test_exploration_type_map_and_atom_map(self):
        data = self.data()
        lines = [
            "atom_style atomic\n",
            "read_data conf.lmp\n",
            "pair_style deepmd graph.000.pt2\n",
            "pair_coeff * *\n",
        ]
        lines = revise_lmp_input_atom_modify(lines, data)
        lines = revise_lmp_input_pair_coeff(lines, data)
        result = "".join(lines)
        self.assertIn("atom_modify", result)
        self.assertIn("map yes", result)
        self.assertIn("pair_coeff      * * Cu", result)
        self.assertLess(result.index("atom_modify"), result.index("read_data"))
