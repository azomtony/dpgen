"""Two extra logs in the run directory, next to ``dpgen.log``.

``dpgen-live.log``
    One short line per check while jobs run: the step, nodes (busy GPUs, time
    left) and job counts. It follows the check interval, so it shows the
    workflow is alive. It is rotated at 20 MB (one ``.1`` copy kept).

``dpgen-summary.log``
    Milestones only: each step finished (and how long it took), exploration
    results per system (accurate / candidate / failed, FP tasks chosen), FP
    progress (25 %, 50 %, 75 %, done, with failures by reason), labels added,
    and nodes started or ended. One screen per iteration.

``dpgen-summary.jsonl``
    The same milestones as JSON lines, for plotting or scripts.

``dpgen.log`` keeps everything, as before.
"""

import json
import os
import time
from pathlib import Path

LIVE = "dpgen-live.log"
SUMMARY = "dpgen-summary.log"
RECORD = "dpgen-summary.jsonl"
LIVE_MAX_BYTES = 20 * 1024 * 1024

STEP_NAMES = (
    "make_train",
    "run_train",
    "post_train",
    "make_model_devi",
    "run_model_devi",
    "post_model_devi",
    "make_fp",
    "run_fp",
    "post_fp",
)

clock = time.time

# Statistics gathered during a step and reported when it finishes.
_notes = {}


def _stamp(now=None):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now or time.time()))


def duration(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours} h {minutes:02d} min"
    days, hours = divmod(hours, 24)
    return f"{days} d {hours} h"


def iter_label(index):
    return f"iter.{int(index):06d}"


def _append(path, text):
    try:
        with open(path, "a") as stream:
            stream.write(text)
    except OSError:
        pass  # logging must never stop the workflow


def live(workdir, message, now=None):
    """One line in dpgen-live.log (call once per check)."""
    path = Path(workdir) / LIVE
    try:
        if path.stat().st_size > LIVE_MAX_BYTES:
            os.replace(path, path.with_name(LIVE + ".1"))
    except OSError:
        pass
    _append(path, f"{_stamp(now)}  {message}\n")


def summary(workdir, event, lines, now=None, **data):
    """Write a milestone to dpgen-summary.log and dpgen-summary.jsonl."""
    if isinstance(lines, str):
        lines = [lines]
    stamp = _stamp(now)
    text = f"{stamp}  {lines[0]}\n" + "".join(
        f"{' ' * (len(stamp) + 2)}{line}\n" for line in lines[1:]
    )
    _append(Path(workdir) / SUMMARY, text)
    record = {"time": now or time.time(), "event": event, **data}
    _append(Path(workdir) / RECORD, json.dumps(record, default=str) + "\n")


# ---- statistics collected during a step ---------------------------------


def note_selection(iteration, system, counts, fp_tasks):
    """Exploration result for one system, reported when make_fp finishes."""
    _notes.setdefault(("selection", int(iteration)), {})[str(system)] = {
        "accurate": int(counts.get("accurate", 0)),
        "candidate": int(counts.get("candidate", 0)),
        "failed": int(counts.get("failed", 0)),
        "fp_tasks": int(fp_tasks),
    }


def note_labels(iteration, total, failed):
    """FP results converted to training data, reported when post_fp finishes."""
    _notes[("labels", int(iteration))] = {"total": int(total), "failed": int(failed)}


def _selection_lines(rows):
    lines = ["  system   accurate  candidate    failed    structures  fp tasks"]
    totals = {"accurate": 0, "candidate": 0, "failed": 0, "fp_tasks": 0}
    for system, row in sorted(rows.items()):
        count = row["accurate"] + row["candidate"] + row["failed"]
        for key in totals:
            totals[key] += row[key]
        lines.append(_selection_row(system, row, count))
    count = totals["accurate"] + totals["candidate"] + totals["failed"]
    if len(rows) > 1:
        lines.append(_selection_row("total", totals, count))
    return lines, totals, count


def _selection_row(name, row, count):
    def pct(value):
        return f"{100 * value / count:5.1f}%" if count else "    -"

    return (
        f"  {name:<7} {pct(row['accurate']):>9}  {pct(row['candidate']):>9} "
        f"{pct(row['failed']):>9}  {count:>11}  {row['fp_tasks']:>8}"
    )


def step_done(workdir, iteration, step, seconds, now=None):
    """A workflow step finished: one line, plus its statistics if any."""
    name = STEP_NAMES[step] if 0 <= step < len(STEP_NAMES) else f"step {step}"
    head = f"{iter_label(iteration)}  {step:02d} {name:<16} done in {duration(seconds)}"
    lines, data = [head], {}
    selection = _notes.pop(("selection", int(iteration)), None)
    if name == "make_fp" and selection is not None:
        table, totals, count = _selection_lines(selection)
        lines += table
        data.update(selection=selection, structures=count, **totals)
    labels = _notes.pop(("labels", int(iteration)), None)
    if name == "post_fp" and labels is not None:
        added = labels["total"] - labels["failed"]
        lines.append(
            f"  labels added: {added} of {labels['total']} FP results "
            f"({labels['failed']} not usable)"
        )
        data.update(labels_added=added, **labels)
    summary(
        workdir,
        "step_done",
        lines,
        now=now,
        iteration=int(iteration),
        step=int(step),
        name=name,
        seconds=round(seconds, 1),
        **data,
    )
