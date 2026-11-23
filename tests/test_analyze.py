import argparse
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
__package__ = "tests"

from dpgen.tools import analyze  # noqa: E402

HEADER = "# step max_devi_v min_devi_v avg_devi_v max_devi_f min_devi_f avg_devi_f\n"


def write_devi(task, rows, frames):
    task.mkdir(parents=True)
    lines = [HEADER] + [
        f"{step} 0.01 0.0 0.008 {f} {fmin} 0.01\n" for step, f, fmin in rows
    ]
    (task / "model_devi.out").write_text("".join(lines))
    (task / "frames.json").write_text(json.dumps(frames))
    (task / "job.json").write_text(json.dumps({"temps": 300}))


def write_system(path, frames):
    (path / "set.000").mkdir(parents=True)
    (path / "type.raw").write_text("0\n")
    np.save(path / "set.000" / "box.npy", np.zeros((frames, 9)))


def read_csv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


class TestAnalyze(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = root = Path(self.tmp.name)
        (root / "param.json").write_text(
            json.dumps(
                {
                    "model_devi_f_trust_lo": 0.05,
                    "model_devi_f_trust_hi": 0.3,
                    "model_devi_jobs": [{}, {"model_devi_f_trust_lo": 0.04}],
                }
            )
        )
        frames = {
            "1000": {"kind": "eq", "event": 0, "time": 1.0, "since_arrival": None},
            "2000": {"kind": "ion", "event": 1, "time": 2.0, "since_arrival": 0.0},
            "2010": {"kind": "ion", "event": 1, "time": 2.01, "since_arrival": 0.03},
            "2020": {"kind": "neutral", "event": 2, "time": 2.02, "since_arrival": 0.3},
        }
        # iter 0: one accurate, one isolated projectile, one candidate, one failed
        write_devi(
            root / "iter.000000/01.model_devi/task.000.000000",
            [
                (1000, 0.02, 0.001),
                (2000, 0.03, 0.0),
                (2010, 0.06, 0.002),
                (2020, 0.5, 0.002),
            ],
            frames,
        )
        # iter 1: trust_lo lowered to 0.04, so 0.045 becomes a candidate
        write_devi(
            root / "iter.000001/01.model_devi/task.000.000000",
            [
                (1000, 0.02, 0.001),
                (2000, 0.02, 0.0),
                (2010, 0.045, 0.002),
                (2020, 0.03, 0.002),
            ],
            frames,
        )
        for it, frames_count in ((0, 10), (1, 30)):
            data = root / f"iter.{it:06d}/00.train/data"
            write_system(data / "sys", frames_count)
            for model in range(2):
                task = root / f"iter.{it:06d}/00.train/{model:03d}"
                task.mkdir(parents=True)
                (task / "input.json").write_text(
                    json.dumps(
                        {
                            "training": {
                                "numb_steps": 1000,
                                "training_data": {"systems": ["../data"]},
                            }
                        }
                    )
                )
                # still falling at the end in iter 1; flat in iter 0
                slope = 0.0 if it == 0 else 1.0
                rows = [
                    f"{s} 1 1 0.01 0.01 {0.1 + slope * (1000 - s) / 1e4:.5f} 0.05 1e-4\n"
                    for s in range(0, 1001, 50)
                ]
                header = "#  step rmse_val rmse_trn rmse_e_val rmse_e_trn rmse_f_val rmse_f_trn lr\n"
                # a restart re-writes the last two steps
                (task / "lcurve.out").write_text(
                    header + "".join(rows[:-2] + rows[-4:])
                )
        fp = root / "iter.000001/02.fp"
        for name in ("task.000.000000", "task.000.000001", "task.000.000002"):
            (fp / name).mkdir(parents=True)
        (fp / "fp-campaign.json").write_text(
            json.dumps(
                {
                    "tasks": {
                        "task.000.000000": {"status": "done"},
                        "task.000.000001": {
                            "status": "stopped",
                            "reason": "SCF stagnation over 60 steps: ...",
                        },
                        "task.000.000002": {"status": "done"},
                    }
                }
            )
        )
        with open(fp / "plasma-origins.000.csv", "w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["fp_task", "kind", "assignment", "max_devi_f"]
            )
            writer.writeheader()
            for name, devi in (
                ("task.000.000000", 0.05),
                ("task.000.000001", 0.2),
                ("task.000.000002", 0.07),
            ):
                writer.writerow(
                    {
                        "fp_task": f"iter.000001/02.fp/{name}",
                        "kind": "ion",
                        "assignment": "training",
                        "max_devi_f": devi,
                    }
                )

    def tearDown(self):
        self.tmp.cleanup()

    def run_part(self, part, **extra):
        args = argparse.Namespace(
            PART=part,
            JOB_DIR=str(self.root),
            param="param.json",
            output="analysis",
            iters=None,
            frames=False,
            no_plots=True,
        )
        vars(args).update(extra)
        return analyze.analyze(args)

    def test_iteration_selection(self):
        self.assertEqual(analyze.iterations(self.root), [0, 1])
        self.assertEqual(analyze.iterations(self.root, "1"), [1])
        self.assertEqual(analyze.iterations(self.root, "0-5"), [0, 1])

    def test_devi_classes_follow_each_iterations_trust_levels(self):
        self.run_part("devi", frames=True)
        out = self.root / "analysis"
        summary = {
            (r["iter"], r["kind"]): r for r in read_csv(out / "devi_summary.csv")
        }
        first = summary[("0", "all")]
        self.assertEqual(float(first["accurate_pct"]), 50.0)
        self.assertEqual(float(first["candidate_pct"]), 25.0)
        self.assertEqual(float(first["failed_pct"]), 25.0)
        self.assertEqual(float(first["isolated_atom_pct"]), 25.0)
        # the per-iteration trust_lo override applies to iter 1 only
        self.assertEqual(float(summary[("1", "all")]["f_trust_lo"]), 0.04)
        self.assertEqual(float(summary[("1", "all")]["candidate_pct"]), 25.0)
        self.assertIn(("1", "ion"), summary)
        frames = read_csv(out / "devi_frames.csv")
        self.assertEqual(len(frames), 8)
        arrival = read_csv(out / "devi_arrival.csv")
        in_flight = [
            r
            for r in arrival
            if r["iter"] == "0" and float(r["since_arrival_to_ps"]) <= 1e-9
        ]
        self.assertEqual(in_flight[0]["frames"], "1")

    def test_train_reads_restarted_lcurve_and_flags_unfinished_training(self):
        notes = self.run_part("train")
        rows = read_csv(self.root / "analysis" / "train_summary.csv")
        self.assertEqual(len(rows), 4)
        latest = [r for r in rows if r["iter"] == "1"]
        self.assertEqual(latest[0]["steps_done"], "1000")
        self.assertEqual(latest[0]["train_frames"], "30")
        self.assertLess(float(latest[0]["rmse_f_val_last10pct_change"]), -0.03)
        self.assertTrue(any("iter 1" in n and "more steps" in n for n in notes))
        self.assertFalse(any("iter 0" in n and "more steps" in n for n in notes))
        self.assertTrue(any("steps per training frame 100.0" in n for n in notes))

    def test_fp_reports_dropped_high_deviation_frames(self):
        notes = self.run_part("fp")
        summary = read_csv(self.root / "analysis" / "fp_summary.csv")
        self.assertEqual(summary[0]["done"], "2")
        self.assertEqual(summary[0]["stopped"], "1")
        self.assertIn("scf stagnation=1", summary[0]["lost_reasons"])
        self.assertTrue(any("hardest frames are being dropped" in n for n in notes))

    @unittest.skipIf(analyze._pyplot() is None, "matplotlib is not installed")
    def test_all_with_plots(self):
        self.run_part("all", no_plots=False)
        out = self.root / "analysis"
        for name in (
            "devi_shares.png",
            "devi_cdf.png",
            "devi_arrival.png",
            "train_rmse.png",
            "train_lcurve.png",
            "fp_status.png",
            "summary.txt",
        ):
            self.assertTrue((out / name).stat().st_size > 0, name)


if __name__ == "__main__":
    unittest.main()
