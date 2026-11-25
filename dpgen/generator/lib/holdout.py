"""Hold out part of each iteration's FP single points as a test set.

With ``fp_holdout_fraction`` set, every new FP task is held out with that
probability (seeded per iteration and system), but each system keeps at least
one task for training. Held-out tasks get a marker file; ``post_fp`` then
writes their labels to ``02.fp/validation.SSS`` instead of ``data.SSS``, so
no model ever trains on them. ``dpgen analyze parity`` merges these sets from
all iterations into one test set.

Plasma jobs keep their own event-balanced holdout and are not marked here.
"""

import json
import math
import os
from collections import defaultdict

import numpy as np

from dpgen import dlog

MARKER = "fp-holdout"


def holdout_options(jdata):
    fraction = jdata.get("fp_holdout_fraction", 0.0)
    seed = jdata.get("fp_holdout_seed", 42)
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 0.5
    ):
        raise ValueError("fp_holdout_fraction must be in [0, 0.5]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("fp_holdout_seed must be a non-negative integer")
    return float(fraction), seed


def is_held_out(task_dir):
    return os.path.isfile(os.path.join(task_dir, MARKER))


def mark_fp_holdout(fp_tasks, iter_index, jdata):
    """Mark a seeded share of ``fp_tasks`` as test-only. Returns marked tasks."""
    fraction, seed = holdout_options(jdata)
    if not fraction or not fp_tasks:
        return []
    if jdata.get("fp_style", "vasp") != "vasp":
        raise ValueError("fp_holdout_fraction is supported with fp_style vasp only")
    jobs = jdata.get("model_devi_jobs", [])
    if iter_index < len(jobs) and jobs[iter_index].get("ensemble") == "plasma":
        dlog.info("FP holdout: plasma jobs use their own validation selection")
        return []
    by_system = defaultdict(list)
    for task in sorted(fp_tasks):
        by_system[int(os.path.basename(task).split(".")[1])].append(task)
    marked = []
    for system, tasks in sorted(by_system.items()):
        rng = np.random.default_rng([seed, iter_index, system])
        chosen = rng.random(len(tasks)) < fraction
        if chosen.all():
            # Keep at least one new label for training.
            chosen[rng.integers(len(tasks))] = False
        for task, held in zip(tasks, chosen):
            if held:
                with open(os.path.join(task, MARKER), "w") as stream:
                    json.dump({"fp_holdout_fraction": fraction, "seed": seed}, stream)
                marked.append(task)
        dlog.info(
            "FP holdout: system %03d keeps %d of %d new single points for testing",
            system,
            int(chosen.sum()),
            len(tasks),
        )
    return marked
