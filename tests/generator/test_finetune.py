#!/usr/bin/env python3

import os
import tempfile
import unittest

from dpgen.generator.finetune import _get_finetune_args, prepare_finetune_jdata


class TestFinetune(unittest.TestCase):
    def test_prepare_finetune_jdata_uses_pytorch_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models = []
            for ii in range(4):
                model = os.path.join(tmpdir, f"model.{ii:03d}.pth")
                with open(model, "w"):
                    pass
                models.append(model)

            jdata = {
                "numb_models": 4,
                "finetune_model": models,
            }

            prepared = prepare_finetune_jdata(jdata)

        self.assertEqual(prepared["train_backend"], "pytorch")
        self.assertEqual(prepared["training_finetune_model"], models)
        self.assertEqual(prepared["finetune_model_source"], "previous")
        self.assertEqual(prepared["training_reuse_iter"], 1)
        self.assertNotIn("training_init_model", prepared)

    def test_prepare_finetune_jdata_adds_use_pretrain_script_for_empty_model(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as model:
            jdata = {
                "numb_models": 1,
                "finetune_model": model.name,
                "default_training_param": {
                    "model": {
                        "type_map": ["C", "H", "N", "O", "S"],
                        "descriptor": {},
                        "fitting_net": {},
                    }
                },
            }

            prepared = prepare_finetune_jdata(jdata)

        self.assertIn("--use-pretrain-script", prepared["finetune_args"])

    def test_prepare_finetune_jdata_preserves_existing_finetune_args(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as model:
            jdata = {
                "numb_models": 1,
                "finetune_model": model.name,
                "finetune_args": "--model-branch base",
                "default_training_param": {
                    "model": {
                        "descriptor": {},
                    }
                },
            }

            prepared = prepare_finetune_jdata(jdata)

        self.assertEqual(
            prepared["finetune_args"], "--model-branch base --use-pretrain-script"
        )

    def test_finetune_model_branch_is_only_used_for_finetune(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as model:
            jdata = {
                "numb_models": 1,
                "finetune_model": model.name,
                "finetune_model_branch": "Omat24",
                "default_training_param": {
                    "model": {
                        "descriptor": {},
                    }
                },
            }

            prepared = prepare_finetune_jdata(jdata)

        self.assertIn("--model-branch Omat24", _get_finetune_args(prepared, True))
        self.assertNotIn("--model-branch Omat24", _get_finetune_args(prepared, False))

    def test_prepare_finetune_jdata_rejects_non_pytorch_backend(self):
        with tempfile.NamedTemporaryFile(suffix=".pb") as model:
            jdata = {
                "numb_models": 1,
                "train_backend": "tensorflow",
                "training_finetune_model": [model.name],
            }

            with self.assertRaisesRegex(RuntimeError, "train_backend='pytorch'"):
                prepare_finetune_jdata(jdata)

    def test_prepare_finetune_jdata_accepts_single_model_alias(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as model:
            jdata = {
                "numb_models": 1,
                "finetune_model": model.name,
            }

            prepared = prepare_finetune_jdata(jdata)

        self.assertEqual(prepared["training_finetune_model"], [model.name])

    def test_prepare_finetune_jdata_accepts_foundation_source(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as model:
            jdata = {
                "numb_models": 1,
                "finetune_model": model.name,
                "finetune_model_source": "foundation",
            }

            prepared = prepare_finetune_jdata(jdata)

        self.assertEqual(prepared["finetune_model_source"], "foundation")

    def test_prepare_finetune_jdata_rejects_model_count_mismatch(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as model:
            jdata = {
                "numb_models": 2,
                "training_finetune_model": [model.name],
            }

            with self.assertRaisesRegex(RuntimeError, "numb_models"):
                prepare_finetune_jdata(jdata)

    def test_prepare_finetune_jdata_rejects_unknown_source(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as model:
            jdata = {
                "numb_models": 1,
                "finetune_model": model.name,
                "finetune_model_source": "latest",
            }

            with self.assertRaisesRegex(RuntimeError, "finetune_model_source"):
                prepare_finetune_jdata(jdata)


if __name__ == "__main__":
    unittest.main()
