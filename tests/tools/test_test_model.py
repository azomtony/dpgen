import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dpgen.tools.test_model import (
    parse_dp_test_output,
    run_model_tests,
    submit_model_tests,
)


class TestModelTest(unittest.TestCase):
    def test_parse_dp_test_output(self):
        metrics = parse_dp_test_output(
            """
            # number of test data : 12
            Energy RMSE/Natoms : 1.25e-03 eV
            Force  RMSE        : 2.5e-02 eV/A
            """
        )
        self.assertEqual(metrics["nframes"], 12)
        self.assertAlmostEqual(metrics["energy_rmse_mev_per_atom"], 1.25)
        self.assertAlmostEqual(metrics["force_rmse_mev_per_angstrom"], 25.0)

    def test_run_model_tests_writes_per_system_outputs(self):
        with tempfile.TemporaryDirectory() as work_dir:
            work_path = Path(work_dir)
            systems_dir = work_path / "systems"
            model_dir = work_path / "models"
            output_dir = work_path / "out"
            fake_bin = work_path / "dp"
            fake_backend_bin = work_path / "fake-dp-pt"

            (systems_dir / "sys.000" / "Cu108O0").mkdir(parents=True)
            (systems_dir / "sys.001").mkdir(parents=True)
            (systems_dir / "sys.000" / "Cu108O0" / "type.raw").write_text("0\n")
            (systems_dir / "sys.001" / "type.raw").write_text("0\n")
            model_dir.mkdir()
            (model_dir / "graph.000.pb").write_text("model")
            fake_bin.write_text(
                "#!/bin/sh\n"
                "echo '# number of test data : 7'\n"
                "echo 'Energy RMSE/Natoms : 2.0e-03 eV'\n"
                "echo 'Force  RMSE        : 3.0e-02 eV/A'\n"
            )
            fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IXUSR)
            fake_backend_bin.write_text(
                "#!/bin/sh\n"
                "test \"$1\" = \"--pt\" || exit 2\n"
                "echo '# number of test data : 7'\n"
                "echo 'Energy RMSE/Natoms : 2.0e-03 eV'\n"
                "echo 'Force  RMSE        : 3.0e-02 eV/A'\n"
            )
            fake_backend_bin.chmod(fake_backend_bin.stat().st_mode | stat.S_IXUSR)

            rows = run_model_tests(
                systems_dir,
                model_dir,
                output_dir,
                dp_command=str(fake_bin),
            )

            self.assertEqual(len(rows), 2)
            self.assertTrue((output_dir / "sys.000" / "dp_test.log").is_file())
            self.assertTrue((output_dir / "sys.001" / "dp_test.log").is_file())
            summary = (output_dir / "summary.tsv").read_text()
            self.assertIn("energy_rmse_mev_per_atom", summary)
            self.assertIn("2", summary)
            self.assertIn("30", summary)

            run_model_tests(
                systems_dir,
                model_dir,
                work_path / "out_pt",
                dp_command=f"{fake_backend_bin} --pt",
            )

    def test_submit_model_tests_uses_dispatcher_layout(self):
        with tempfile.TemporaryDirectory() as work_dir:
            work_path = Path(work_dir)
            output_dir = work_path / "out"
            machine_file = work_path / "machine.json"
            system = work_path / "sys.000"
            model = work_path / "graph.000.pth"
            system.mkdir()
            (system / "type.raw").write_text("0\n")
            model.write_text("model")
            machine_file.write_text(
                """
                {
                  "api_version": "1.0",
                  "train": {
                    "command": "dp --pt",
                    "machine": {"local_root": "./"},
                    "resources": {"group_size": 4}
                  }
                }
                """
            )
            case = {
                "system": system,
                "system_key": "sys.000",
                "system_name": "sys.000",
                "model": model,
                "run_task": "sys.000",
                "run_dir": output_dir / "sys.000",
            }
            fake_submission = mock.Mock()

            with mock.patch(
                "dpgen.tools.test_model._make_submission",
                return_value=fake_submission,
            ) as make_submission:
                submit_model_tests([case], output_dir, machine_file, "train")

            self.assertTrue((output_dir / "sys.000" / "system").is_symlink())
            self.assertTrue((output_dir / "sys.000" / "model.pth").is_symlink())
            make_submission.assert_called_once()
            kwargs = make_submission.call_args.kwargs
            self.assertEqual(
                kwargs["commands"],
                ["( cd sys.000 && dp --pt test -m model.pth -s system -n 0 -d detail > dp_test.log 2>&1 )"],
            )
            self.assertEqual(kwargs["run_tasks"], ["."])
            self.assertEqual(kwargs["group_size"], 4)
            self.assertEqual(kwargs["forward_files"], ["sys.000/model.pth", "sys.000/system"])
            fake_submission.run_submission.assert_called_once()


if __name__ == "__main__":
    unittest.main()
