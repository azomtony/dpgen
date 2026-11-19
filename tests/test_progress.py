"""dpgen-live.log and dpgen-summary.log."""

import json
import tempfile
import unittest
from pathlib import Path

from dpgen import progress


class TestProgress(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def records(self):
        return [
            json.loads(line)
            for line in (self.work / progress.RECORD).read_text().splitlines()
        ]

    def test_make_fp_reports_selection_per_system_and_total(self):
        progress.note_selection(
            9, "000", {"accurate": 900, "candidate": 80, "failed": 20}, 40
        )
        progress.note_selection(
            9, "001", {"accurate": 450, "candidate": 30, "failed": 20}, 30
        )
        progress.step_done(self.work, 9, 6, 125)
        text = (self.work / progress.SUMMARY).read_text()
        self.assertIn("iter.000009  06 make_fp", text)
        self.assertIn("done in 2 min", text)
        self.assertIn(
            "000         90.0%       8.0%      2.0%         1000        40", text
        )
        self.assertIn(
            "total       90.0%       7.3%      2.7%         1500        70", text
        )
        record = self.records()[-1]
        self.assertEqual(record["fp_tasks"], 70)
        self.assertEqual(record["candidate"], 110)
        self.assertEqual(record["selection"]["001"]["failed"], 20)

    def test_post_fp_reports_labels_and_other_steps_one_line(self):
        progress.note_labels(9, 70, 3)
        progress.step_done(self.work, 9, 8, 30)
        progress.step_done(self.work, 10, 1, 5 * 3600 + 120)
        text = (self.work / progress.SUMMARY).read_text()
        self.assertIn("labels added: 67 of 70 FP results (3 not usable)", text)
        self.assertIn("iter.000010  01 run_train        done in 5 h 02 min", text)
        self.assertEqual(self.records()[0]["labels_added"], 67)

    def test_live_log_rotates(self):
        for index in range(3):
            progress.live(self.work, f"check {index}")
        lines = (self.work / progress.LIVE).read_text().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[-1].endswith("check 2"))
        old, progress.LIVE_MAX_BYTES = progress.LIVE_MAX_BYTES, 10
        try:
            progress.live(self.work, "after rotation")
        finally:
            progress.LIVE_MAX_BYTES = old
        self.assertTrue((self.work / (progress.LIVE + ".1")).exists())
        self.assertEqual(len((self.work / progress.LIVE).read_text().splitlines()), 1)

    def test_logging_never_raises(self):
        progress.live(self.work / "missing" / "dir", "x")
        progress.summary(self.work / "missing", "e", "x")


if __name__ == "__main__":
    unittest.main()
