import argparse
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

from dpgen.dispatcher.interactive import (
    add_interactive_args,
    interactive_execution,
    local_submission,
)


class TestInteractive(unittest.TestCase):
    def args(self, *argv):
        parser = argparse.ArgumentParser()
        add_interactive_args(parser)
        return parser.parse_args(argv)

    def test_validation_and_batch_default(self):
        with interactive_execution(self.args()):
            self.assertIsNone(local_submission({}, [], ".", [], "log", "log"))
        for argv in [
            ("--gpus-per-job", "2"),
            ("--gpus", "0", "0"),
            ("--gpus", "-1"),
            ("--gpus", "0", "1", "--gpus-per-job", "3"),
            ("--gpus", "0", "--gpus-per-job", "0"),
        ]:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                with interactive_execution(self.args(*argv)):
                    pass

    def test_gpu_pool_and_stage_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup = root / "stage env.sh"
            setup.write_text("export STAGE=dp\nexport CUDA_VISIBLE_DEVICES=wrong\n")
            script = root / "task.py"
            script.write_text("""import json, os, time
from pathlib import Path
start = time.monotonic()
time.sleep(0.15)
Path("result.json").write_text(json.dumps({"start": start, "end": time.monotonic(), "gpus": os.environ["CUDA_VISIBLE_DEVICES"], "stage": os.environ["STAGE"]}))
""")
            tasks = [str(i) for i in range(5)]
            for task in tasks:
                (root / task).mkdir()
            command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
            with interactive_execution(
                self.args("--gpus", "0", "1", "2", "3", "--gpus-per-job", "2")
            ):
                submission = local_submission(
                    {"source_list": [str(setup)]}, [command], root, tasks, "log", "log"
                )
                submission.run_submission()
                results = [
                    json.loads((root / task / "result.json").read_text())
                    for task in tasks
                ]
                self.assertEqual({r["gpus"] for r in results}, {"0,1", "2,3"})
                self.assertEqual({r["stage"] for r in results}, {"dp"})
                for i, first in enumerate(results):
                    for second in results[i + 1 :]:
                        if first["gpus"] == second["gpus"]:
                            self.assertTrue(
                                first["end"] <= second["start"]
                                or second["end"] <= first["start"]
                            )
                self.assertTrue(
                    any(
                        a["start"] < b["end"] and b["start"] < a["end"]
                        for a in results
                        for b in results
                        if a["gpus"] != b["gpus"]
                    )
                )
                local_submission(
                    {"envs": {"STAGE": "vasp"}},
                    [command],
                    root,
                    tasks[:1],
                    "log",
                    "err",
                ).run_submission()
                self.assertEqual(
                    json.loads((root / "0/result.json").read_text())["stage"], "vasp"
                )
            self.assertIsNone(local_submission({}, [], root, [], "log", "log"))

    def test_failure_does_not_run_later_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            with interactive_execution(self.args("--gpus", "0")):
                submission = local_submission(
                    {"append_script": ["touch appended"]},
                    ["false", "touch later"],
                    tmp,
                    ["."],
                    "log",
                    "log",
                )
                with self.assertRaisesRegex(RuntimeError, "exit code 1"):
                    submission.run_submission()
            self.assertFalse((Path(tmp) / "later").exists())
            self.assertFalse((Path(tmp) / "appended").exists())

    def test_failed_peer_is_terminated_and_pending_task_not_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("fail", "peer", "pending"):
                (root / name).mkdir()
            command = (
                'case "$PWD" in */fail) sleep 0.1; exit 7;; '
                "*/peer) sleep 1; touch survived;; *) touch started;; esac"
            )
            with interactive_execution(self.args("--gpus", "0", "1")):
                submission = local_submission(
                    {}, [command], root, ["fail", "peer", "pending"], "log", "log"
                )
                with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                    submission.run_submission()
            self.assertFalse((root / "peer/survived").exists())
            self.assertFalse((root / "pending/started").exists())

    def test_cli_routes_both_workflows_through_local_dispatcher(self):
        from unittest.mock import patch

        from dpgen.dispatcher.Dispatcher import make_submission
        from dpgen.dispatcher.interactive import LocalSubmission
        from dpgen.main import main_parser

        def run_workflow(*args):
            submission = make_submission(
                {"batch_type": "Slurm", "local_root": "/unused"},
                {},
                ["true"],
                ".",
                [],
                99,
                [],
                [],
                [],
                "log",
                "log",
            )
            self.assertIsInstance(submission, LocalSubmission)
            self.assertEqual(submission.groups, [[0], [1]])
            submission.run_submission()

        for command, target in [
            ("run", "dpgen.generator.run.run_iter"),
            ("finetune", "dpgen.generator.finetune.run_finetune_iter"),
        ]:
            with (
                self.subTest(command=command),
                patch(target, side_effect=run_workflow) as run,
            ):
                args = main_parser().parse_args(
                    [command, "param.json", "machine.json", "--gpus", "0", "1"]
                )
                args.func(args)
                run.assert_called_once_with("param.json", "machine.json")
            self.assertIsNone(local_submission({}, [], ".", [], "log", "log"))

    def test_batch_submission_still_uses_dispatcher(self):
        from unittest.mock import patch

        from dpgen.dispatcher.Dispatcher import make_submission

        with (
            patch("dpgen.dispatcher.Dispatcher.Machine") as machine,
            patch("dpgen.dispatcher.Dispatcher.Resources"),
            patch("dpgen.dispatcher.Dispatcher.Submission") as submission,
        ):
            result = make_submission(
                {}, {}, ["true"], ".", [], 1, [], [], [], "log", "log"
            )
            machine.load_from_dict.assert_called_once()
            self.assertIs(result, submission.return_value)

    def test_empty_stage(self):
        with interactive_execution(self.args("--gpus", "0")):
            local_submission({}, [], ".", [], "log", "log").run_submission()
