import subprocess as sp
import unittest


class TestCLI(unittest.TestCase):
    def test_cli(self):
        sp.check_output(["dpgen", "-h"])
        for subcommand in (
            "run",
            "finetune",
            "simplify",
            "init_surf",
            "init_bulk",
            "init_reaction",
            "autotest",
            "test-model",
        ):
            sp.check_output(["dpgen", subcommand, "-h"])
