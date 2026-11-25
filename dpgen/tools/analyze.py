"""Diagnose a DP-GEN run that stops improving.

``dpgen analyze devi|train|fp|all [JOB_DIR]`` reads what a run already wrote and
puts CSV tables, PNG plots (when matplotlib is installed) and ``summary.txt``
into ``JOB_DIR/analysis``:

devi
    Committee force deviation of every explored frame, per iteration, system
    and plasma phase (eq / ion / neutral), against the trust levels in force
    for that iteration. Answers: is the accurate share rising, do candidates
    sit just above ``trust_lo``, and when after an impact the models disagree.
train
    ``lcurve.out`` of every model: final training/validation RMSE, whether
    the validation error was still falling when training stopped, and how the
    training data grew against the steps trained.
fp
    What happened to the frames sent to FP: labelled, stopped by the monitor,
    failed or discarded, and whether the lost ones were the high-deviation
    frames the models most needed.
parity
    Merges every iteration's held-out FP labels (``fp_holdout_fraction``)
    into one test set, runs ``dp test`` with the models of each selected
    iteration (``-i``), and plots model against DFT for energy per atom and
    force, one row per iteration, with RMSE in each panel. Not part of
    ``all``, because it runs ``dp test``; use ``--machine`` to run it as a job.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PARTS = ("devi", "train", "fp")
# Time after a projectile reaches the surface, in ps. The first bin holds the
# frames before impact (since_arrival is clamped to 0 while the atom flies).
ARRIVAL_BINS = (0.0, 1e-9, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, math.inf)
LOST = ("stopped", "failed", "discarded")
# Reference palette, light surface (dataviz skill): categorical slots in fixed
# order, a single blue ramp for iterations, muted ink for thresholds.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# --------------------------------------------------------------------------
# shared helpers


def iterations(root, selection=None):
    found = sorted(
        int(path.name.split(".")[1])
        for path in Path(root).glob("iter.[0-9]*")
        if path.is_dir() and path.name.split(".")[1].isdigit()
    )
    if not selection:
        return found
    wanted = set()
    for part in str(selection).split(","):
        part = part.strip()
        if "-" in part:
            first, last = part.split("-", 1)
            wanted.update(range(int(first), int(last) + 1))
        elif part:
            wanted.add(int(part))
    return [it for it in found if it in wanted]


def per_system(value, system):
    if isinstance(value, dict):
        return float(value.get(str(system), value.get(system)))
    if isinstance(value, (list, tuple)):
        return float(value[system])
    return float(value)


def trust(jdata, iteration, system):
    """Trust levels used to classify this iteration's frames."""
    jobs = jdata.get("model_devi_jobs", [])
    job = jobs[iteration] if iteration < len(jobs) else {}

    def get(key, default):
        return per_system(job.get(key, jdata.get(key, default)), system)

    return (
        get("model_devi_f_trust_lo", math.inf),
        get("model_devi_f_trust_hi", math.inf),
        get("model_devi_v_trust_lo", 1e10),
        get("model_devi_v_trust_hi", 1e10),
    )


def classify(f, v, f_lo, f_hi, v_lo, v_hi):
    """Same rule as frame selection (plasma.select_frames)."""
    if not (math.isfinite(f) and math.isfinite(v)) or f >= f_hi or v >= v_hi:
        return "failed"
    if f >= f_lo or v >= v_lo:
        return "candidate"
    return "accurate"


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def load_table(path):
    """Numeric rows of a whitespace table; tolerate a half-written last line."""
    rows = []
    try:
        with open(path) as stream:
            for line in stream:
                if line.lstrip().startswith("#") or not line.strip():
                    continue
                try:
                    rows.append([float(x) for x in line.split()])
                except ValueError:
                    continue
    except OSError:
        return []
    width = max((len(r) for r in rows), default=0)
    return [r for r in rows if len(r) == width]


def quantiles(values, points=(50, 90, 99)):
    finite = np.asarray([x for x in values if x is not None and math.isfinite(x)])
    if not finite.size:
        return {f"p{p}": None for p in points} | {"max": None}
    result = {f"p{p}": float(np.percentile(finite, p)) for p in points}
    result["max"] = float(finite.max())
    return result


def write_csv(path, rows, columns=None):
    rows = list(rows)
    columns = columns or (list(rows[0]) if rows else [])
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (f"{value:.6g}" if isinstance(value, float) else value)
                    for key, value in row.items()
                }
            )


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


def fmt(value, spec=".4g"):
    return "-" if value is None else format(value, spec)


# --------------------------------------------------------------------------
# devi: committee deviation of explored frames


def collect_frames(root, jdata, iters):
    skip = jdata.get("model_devi_skip", 0)
    frames = []
    for it in iters:
        base = Path(root) / f"iter.{it:06d}" / "01.model_devi"
        for task in sorted(base.glob("task.*")):
            rows = load_table(task / "model_devi.out")
            if not rows or len(rows[0]) < 7:
                continue
            system = int(task.name.split(".")[1])
            levels = trust(jdata, it, system)
            meta = read_json(task / "frames.json", {})
            temp = read_json(task / "job.json", {}).get("temps")
            for row in rows:
                if row[0] < skip:
                    continue
                step = int(row[0])
                frame = meta.get(str(step), {})
                f, v = row[4], row[1]
                frames.append(
                    {
                        "iter": it,
                        "system": system,
                        "task": task.name,
                        "step": step,
                        "kind": frame.get("kind", "md"),
                        "event": frame.get("event"),
                        "time_ps": frame.get("time"),
                        "since_arrival_ps": frame.get("since_arrival"),
                        "temp": temp,
                        "max_devi_f": f,
                        "min_devi_f": row[5],
                        "avg_devi_f": row[6],
                        "max_devi_v": v,
                        # An atom with no neighbours (a projectile still in
                        # flight) gets zero force from every model.
                        "isolated_atom": row[5] == 0.0,
                        "class": classify(f, v, *levels),
                        "f_trust_lo": levels[0],
                        "f_trust_hi": levels[1],
                    }
                )
    return frames


def summarize_frames(frames):
    groups = defaultdict(list)
    for frame in frames:
        groups[(frame["iter"], frame["system"], "all")].append(frame)
        groups[(frame["iter"], frame["system"], frame["kind"])].append(frame)
    rows = []
    for (it, system, kind), values in sorted(groups.items()):
        if kind != "all" and {v["kind"] for v in values} == {"md"}:
            continue  # plain MD: the "all" row already says it
        counts = Counter(v["class"] for v in values)
        lo, hi = values[0]["f_trust_lo"], values[0]["f_trust_hi"]
        candidates = [v["max_devi_f"] for v in values if v["class"] == "candidate"]
        n = len(values)
        rows.append(
            {
                "iter": it,
                "system": system,
                "kind": kind,
                "frames": n,
                "accurate_pct": pct(counts["accurate"], n),
                "candidate_pct": pct(counts["candidate"], n),
                "failed_pct": pct(counts["failed"], n),
                # Candidates that barely pass trust_lo: if most do, trust_lo
                # sits at the model's noise floor rather than at real errors.
                "candidates_below_1.25lo_pct": pct(
                    sum(f < 1.25 * lo for f in candidates), len(candidates)
                ),
                "isolated_atom_pct": pct(sum(v["isolated_atom"] for v in values), n),
                "f_trust_lo": lo,
                "f_trust_hi": hi,
                **quantiles(v["max_devi_f"] for v in values),
            }
        )
    return rows


def arrival_profile(frames):
    rows = []
    groups = defaultdict(list)
    for frame in frames:
        if (
            frame["kind"] in ("ion", "neutral")
            and frame["since_arrival_ps"] is not None
        ):
            groups[(frame["iter"], frame["system"], frame["kind"])].append(frame)
    for (it, system, kind), values in sorted(groups.items()):
        for start, end in zip(ARRIVAL_BINS, ARRIVAL_BINS[1:]):
            inside = [v for v in values if start <= v["since_arrival_ps"] < end]
            if not inside:
                continue
            counts = Counter(v["class"] for v in inside)
            q = quantiles((v["max_devi_f"] for v in inside), (50, 90))
            rows.append(
                {
                    "iter": it,
                    "system": system,
                    "kind": kind,
                    "since_arrival_from_ps": start,
                    "since_arrival_to_ps": end,
                    "frames": len(inside),
                    "median_max_devi_f": q["p50"],
                    "p90_max_devi_f": q["p90"],
                    "candidate_pct": pct(counts["candidate"], len(inside)),
                    "failed_pct": pct(counts["failed"], len(inside)),
                }
            )
    return rows


def devi_findings(summary):
    notes = []
    by_system = defaultdict(list)
    for row in summary:
        if row["kind"] == "all":
            by_system[row["system"]].append(row)
    for system, rows in sorted(by_system.items()):
        if len(rows) < 2:
            continue
        first, last = rows[0], rows[-1]
        trend = last["accurate_pct"] - first["accurate_pct"]
        notes.append(
            f"devi: system {system:03d}: accurate {first['accurate_pct']:.1f}% (iter {first['iter']}) "
            f"-> {last['accurate_pct']:.1f}% (iter {last['iter']}), "
            f"failed {first['failed_pct']:.1f}% -> {last['failed_pct']:.1f}%"
            + ("; little progress" if abs(trend) < 5 else "")
        )
        if last["candidates_below_1.25lo_pct"] > 50:
            notes.append(
                f"devi: system {system:03d}: {last['candidates_below_1.25lo_pct']:.0f}% of the latest "
                f"candidates lie within 25% above trust_lo={last['f_trust_lo']:.3g}; trust_lo may sit "
                "at the committee's noise floor (check `analyze devi` CDF before lowering FP effort there)"
            )
        if (
            last["p50"] is not None
            and first["p50"] is not None
            and last["p50"] >= first["p50"]
        ):
            notes.append(
                f"devi: system {system:03d}: median max_devi_f did not fall "
                f"({first['p50']:.3g} -> {last['p50']:.3g} eV/A)"
            )
    phases = [r for r in summary if r["kind"] in ("eq", "ion", "neutral")]
    if phases:
        latest = max(r["iter"] for r in phases)
        rows = [r for r in phases if r["iter"] == latest]
        worst = max(rows, key=lambda r: r["candidate_pct"] + r["failed_pct"])
        notes.append(
            f"devi: iter {latest}: most uncertain phase is '{worst['kind']}' (system {worst['system']:03d}): "
            f"candidate {worst['candidate_pct']:.1f}%, failed {worst['failed_pct']:.1f}%, "
            f"p90 max_devi_f {fmt(worst['p90'], '.3g')} eV/A"
        )
    return notes


# --------------------------------------------------------------------------
# train: lcurve.out convergence


def read_lcurve(path):
    names, rows = None, []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return None, []
    for line in lines:
        if line.startswith("#"):
            if names is None and "step" in line:
                names = line[1:].split()
            continue
        try:
            values = [float(x) for x in line.split()]
        except ValueError:
            continue
        if not values or (names and len(values) != len(names)):
            continue
        # A restart re-writes steps it had already passed; keep the latest.
        while rows and rows[-1][0] >= values[0]:
            rows.pop()
        rows.append(values)
    return names, rows


def count_frames(directory, cache):
    """Frames in every deepmd/npy system below a data path."""
    directory = Path(directory)
    key = str(directory.resolve()) if directory.exists() else str(directory)
    if key in cache:
        return cache[key]
    total = 0
    for type_file in directory.rglob("type.raw") if directory.is_dir() else []:
        for box in type_file.parent.glob("set.*/box.npy"):
            try:
                total += int(np.load(box, mmap_mode="r").shape[0])
            except (OSError, ValueError):
                pass
    cache[key] = total
    return total


def data_frames(task, section, cache):
    systems = section.get("systems", []) if isinstance(section, dict) else []
    if isinstance(systems, str):
        systems = [systems]
    return sum(count_frames(Path(task) / s, cache) for s in systems)


def lcurve_metrics(names, rows):
    """Robust end-of-training values from a noisy lcurve.

    Each lcurve row is the error of one batch, so single rows swing by a
    factor of several. Medians over a fifth of the run are stable enough to
    compare models and iterations.
    """
    if not rows:
        return {}
    table = np.asarray(rows)
    n = len(table)
    window = max(1, n // 5)
    result = {"steps_done": int(table[-1, 0])}
    for name in ("rmse_f_val", "rmse_f_trn", "rmse_e_val", "rmse_e_trn", "lr"):
        if name not in names:
            continue
        column = table[:, names.index(name)]
        if name == "lr":
            result[name] = float(column[-1])
            continue
        result[name] = float(np.median(column[-window:]))
        if n >= 2 * window:
            before = float(np.median(column[-2 * window : -window]))
            # Change of the median over the last fifth of training against the
            # fifth before; clearly negative = still falling when it stopped.
            result[f"{name}_tail_change"] = (
                (result[name] - before) / before if before else None
            )
    return result


def count_starts(path):
    """Training runs in one lcurve.out: each start writes a block of # lines."""
    starts, previous = 0, False
    try:
        for line in Path(path).read_text().splitlines():
            header = line.startswith("#")
            starts += header and not previous
            previous = header
    except OSError:
        pass
    return starts


def smooth(values, width):
    """Rolling median, same length as the input."""
    values = np.asarray(values, dtype=float)
    half = max(1, width) // 2
    return np.array(
        [np.median(values[max(0, i - half) : i + half + 1]) for i in range(len(values))]
    )


def collect_training(root, iters):
    cache = {}
    rows = []
    curves = {}
    for it in iters:
        base = Path(root) / f"iter.{it:06d}" / "00.train"
        for task in sorted(base.glob("[0-9][0-9][0-9]")):
            names, data = read_lcurve(task / "lcurve.out")
            if not data:
                continue
            names = names or [
                "step",
                "rmse_val",
                "rmse_trn",
                "rmse_e_val",
                "rmse_e_trn",
                "rmse_f_val",
                "rmse_f_trn",
                "lr",
            ]
            config = read_json(task / "input.json", {})
            training = config.get("training", {})
            curves[(it, task.name)] = (names, data)
            rows.append(
                {
                    "iter": it,
                    "model": task.name,
                    "numb_steps": training.get("numb_steps"),
                    "train_frames": data_frames(
                        task, training.get("training_data", {}), cache
                    ),
                    "valid_frames": data_frames(
                        task, training.get("validation_data", {}), cache
                    ),
                    "starts": count_starts(task / "lcurve.out"),
                    **lcurve_metrics(names, data),
                }
            )
    return rows, curves


def train_findings(rows):
    notes = []
    by_iter = defaultdict(list)
    for row in rows:
        by_iter[row["iter"]].append(row)
    if not by_iter:
        return ["train: no lcurve.out found"]
    for it in sorted(by_iter)[-3:]:
        models = by_iter[it]
        key = "rmse_f_val" if "rmse_f_val" in models[0] else "rmse_f_trn"
        values = [m[key] for m in models if m.get(key) is not None]
        if not values:
            continue
        changes = [m.get(f"{key}_tail_change") for m in models]
        changes = [c for c in changes if c is not None]
        line = (
            f"train: iter {it}: {key} {min(values):.4g}-{max(values):.4g} eV/A over {len(values)} models, "
            f"{models[0].get('train_frames', 0)} training frames, numb_steps {models[0].get('numb_steps')}"
        )
        if changes:
            change = float(np.median(changes))
            if change < -0.05:
                line += f"; still falling {-100 * change:.0f}% over the last fifth -> more steps should help"
            elif abs(change) <= 0.05:
                line += "; flat over the last fifth -> more steps alone will not help"
        notes.append(line)
        starts = max(m.get("starts", 1) for m in models)
        if starts > 1:
            notes.append(
                f"train: iter {it}: training was started {starts} times (restarts after lost nodes)"
            )
        # A model far off the others inflates the committee deviation everywhere.
        for name, unit in ((key, "eV/A"), (key.replace("_f_", "_e_"), "eV")):
            finals = {m["model"]: m.get(name) for m in models if m.get(name)}
            for model, value in finals.items():
                others = [v for k, v in finals.items() if k != model]
                if len(others) >= 2 and value > 2 * np.median(others):
                    notes.append(
                        f"train: iter {it}: model {model} {name} {value:.3g} {unit} is "
                        f"{value / np.median(others):.1f}x the other models' median; "
                        "it will inflate model deviation for every frame"
                    )
        trn = [m.get("rmse_f_trn") for m in models if m.get("rmse_f_trn")]
        val = [m.get("rmse_f_val") for m in models if m.get("rmse_f_val")]
        if trn and val and np.median(val) > 1.5 * np.median(trn):
            notes.append(
                f"train: iter {it}: validation force RMSE is {np.median(val) / np.median(trn):.1f}x training; "
                "the models fit the training frames much better than unseen ones"
            )
        elif trn and not val:
            notes.append(
                f"train: iter {it}: no validation_data in input.json, so only training error is known; "
                "use `dp test` on the 02.fp/validation.* frames for held-out error"
            )
    iters = sorted(by_iter)
    if len(iters) >= 2:
        first, last = by_iter[iters[0]][0], by_iter[iters[-1]][0]
        if (
            first.get("train_frames")
            and first.get("numb_steps")
            and last.get("numb_steps")
        ):
            ratio_first = first["numb_steps"] / first["train_frames"]
            ratio_last = last["numb_steps"] / max(1, last["train_frames"])
            notes.append(
                f"train: steps per training frame {ratio_first:.1f} (iter {iters[0]}) -> {ratio_last:.1f} (iter {iters[-1]})"
                + (
                    "; each frame is seen far less often now"
                    if ratio_last < 0.5 * ratio_first
                    else ""
                )
            )
    return notes


# --------------------------------------------------------------------------
# fp: what became a label


def stop_kind(reason):
    reason = (reason or "").lower()
    for word, label in (
        ("stagnation", "scf stagnation"),
        ("iteration limit", "scf step limit"),
        ("stall", "stall timeout"),
        ("startup", "startup timeout"),
        ("nonfinite", "nonfinite energy"),
        ("wall-time", "wall-time budget"),
    ):
        if word in reason:
            return label
    return reason[:40] or "-"


def collect_fp(root, iters):
    rows = []
    for it in iters:
        base = Path(root) / f"iter.{it:06d}" / "02.fp"
        if not base.is_dir():
            continue
        origins = {}
        for path in base.glob("plasma-origins.*.csv"):
            with open(path, newline="") as stream:
                for origin in csv.DictReader(stream):
                    origins[Path(origin["fp_task"]).name] = origin
        tasks = (read_json(base / "fp-campaign.json", {}) or {}).get("tasks", {})
        names = sorted(set(tasks) | {p.name for p in base.glob("task.*") if p.is_dir()})
        for name in names:
            entry = tasks.get(name)
            if entry is None:
                task = base / name
                status = (
                    "stopped"
                    if (task / "fp-monitor-stop.json").exists()
                    else "done"
                    if (task / "OUTCAR").exists()
                    else "unknown"
                )
                reason = (read_json(task / "fp-monitor-stop.json", {}) or {}).get(
                    "reason", ""
                )
            else:
                status, reason = entry.get("status", "unknown"), entry.get("reason", "")
            origin = origins.get(name, {})
            devi = origin.get("max_devi_f")
            rows.append(
                {
                    "iter": it,
                    "task": name,
                    "status": status,
                    "why": stop_kind(reason) if status in LOST else "",
                    "reason": reason,
                    "kind": origin.get("kind", ""),
                    "assignment": origin.get("assignment", ""),
                    "max_devi_f": float(devi)
                    if devi not in (None, "", "None")
                    else None,
                }
            )
    return rows


def summarize_fp(rows):
    out = []
    by_iter = defaultdict(list)
    for row in rows:
        by_iter[row["iter"]].append(row)
    for it, values in sorted(by_iter.items()):
        counts = Counter(v["status"] for v in values)
        done = [
            v["max_devi_f"]
            for v in values
            if v["status"] == "done" and v["max_devi_f"] is not None
        ]
        lost = [
            v["max_devi_f"]
            for v in values
            if v["status"] in LOST and v["max_devi_f"] is not None
        ]
        why = Counter(v["why"] for v in values if v["status"] in LOST)
        out.append(
            {
                "iter": it,
                "tasks": len(values),
                "done": counts["done"],
                "stopped": counts["stopped"],
                "failed": counts["failed"],
                "discarded": counts["discarded"],
                "other": len(values) - sum(counts[s] for s in ("done", *LOST)),
                "labelled_pct": pct(counts["done"], len(values)),
                "median_devi_done": float(np.median(done)) if done else None,
                "median_devi_lost": float(np.median(lost)) if lost else None,
                "lost_reasons": "; ".join(f"{k}={n}" for k, n in why.most_common()),
            }
        )
    return out


def fp_findings(summary):
    notes = []
    for row in summary:
        lost = row["stopped"] + row["failed"] + row["discarded"]
        if not lost:
            continue
        line = f"fp: iter {row['iter']}: {lost}/{row['tasks']} frames not labelled ({row['lost_reasons']})"
        if row["median_devi_lost"] and row["median_devi_done"]:
            ratio = row["median_devi_lost"] / row["median_devi_done"]
            line += f"; their median max_devi_f is {ratio:.2f}x the labelled ones"
            if ratio > 1.2:
                line += " -> the hardest frames are being dropped"
        notes.append(line)
    return notes or ["fp: every FP task in the selected iterations was labelled"]


# --------------------------------------------------------------------------
# plots (optional)


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "savefig.dpi": 150,
        }
    )
    return plt


def integer_ticks(ax):
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def blues(n):
    """Sequential single-hue ramp, light (early) to dark (late)."""
    light, dark = np.array([0xB5, 0xD0, 0xF2]) / 255, np.array([0x0D, 0x36, 0x6E]) / 255
    if n == 1:
        return [tuple(dark)]
    return [tuple(light + (dark - light) * i / (n - 1)) for i in range(n)]


def _systems(rows):
    return sorted({r["system"] for r in rows})


def plot_devi(plt, frames, summary, arrival, out):
    systems = _systems(summary)
    if not systems:
        return
    # 1. accurate / candidate / failed share per iteration
    fig, axes = plt.subplots(
        1, len(systems), figsize=(4.5 * len(systems), 3.4), squeeze=False
    )
    for ax, system in zip(axes[0], systems):
        rows = [r for r in summary if r["system"] == system and r["kind"] == "all"]
        x = [r["iter"] for r in rows]
        for i, name in enumerate(("accurate", "candidate", "failed")):
            y = [r[f"{name}_pct"] for r in rows]
            ax.plot(x, y, marker="o", markersize=4, color=SERIES[i], label=name)
            ax.annotate(
                name,
                (x[-1], y[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                color=INK,
                va="center",
            )
        ax.set(
            title=f"System {system:03d}: frame classes",
            xlabel="iteration",
            ylabel="share of frames (%)",
            ylim=(0, 100),
        )
        integer_ticks(ax)
    axes[0][0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "devi_shares.png")
    plt.close(fig)

    # 2. distribution of max_devi_f per iteration (CDF), trust levels marked
    fig, axes = plt.subplots(
        1, len(systems), figsize=(4.5 * len(systems), 3.4), squeeze=False
    )
    for ax, system in zip(axes[0], systems):
        values = defaultdict(list)
        for frame in frames:
            if (
                frame["system"] == system
                and math.isfinite(frame["max_devi_f"])
                and frame["max_devi_f"] > 0
            ):
                values[frame["iter"]].append(frame["max_devi_f"])
        iters = sorted(values)
        for color, it in zip(blues(len(iters)), iters):
            data = np.sort(values[it])
            ax.plot(
                data,
                np.arange(1, len(data) + 1) / len(data),
                color=color,
                linewidth=1.5,
                label=f"iter {it}",
            )
        rows = [r for r in summary if r["system"] == system and r["kind"] == "all"]
        if rows:
            for level, name in (
                (rows[-1]["f_trust_lo"], "trust_lo"),
                (rows[-1]["f_trust_hi"], "trust_hi"),
            ):
                if math.isfinite(level):
                    ax.axvline(level, color=MUTED, linestyle="--", linewidth=1)
                    ax.annotate(
                        name,
                        (level, 0.98),
                        xytext=(-3, 0),
                        ha="right",
                        va="top",
                        textcoords="offset points",
                        fontsize=8,
                        color=MUTED,
                    )
        ax.set_xscale("log")
        from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set(
            title=f"System {system:03d}: max force deviation",
            xlabel="max_devi_f (eV/Å)",
            ylabel="fraction of frames below",
            ylim=(0, 1),
        )
        if iters:
            ax.legend(loc="lower right", ncol=2 if len(iters) > 6 else 1)
    fig.tight_layout()
    fig.savefig(out / "devi_cdf.png")
    plt.close(fig)

    # 3. deviation against time since impact, latest iteration
    if arrival:
        latest = max(r["iter"] for r in arrival)
        fig, axes = plt.subplots(
            1, len(systems), figsize=(4.5 * len(systems), 3.4), squeeze=False
        )
        for ax, system in zip(axes[0], systems):
            for i, kind in enumerate(("ion", "neutral")):
                rows = [
                    r
                    for r in arrival
                    if r["iter"] == latest
                    and r["system"] == system
                    and r["kind"] == kind
                ]
                if not rows:
                    continue
                # One fixed x position per time bin, shared by both kinds.
                x = [ARRIVAL_BINS.index(r["since_arrival_from_ps"]) for r in rows]
                ax.plot(
                    x,
                    [r["median_max_devi_f"] for r in rows],
                    marker="o",
                    markersize=4,
                    color=SERIES[i],
                    label=f"{kind} median",
                )
                ax.plot(
                    x,
                    [r["p90_max_devi_f"] for r in rows],
                    marker="o",
                    markersize=4,
                    color=SERIES[i],
                    linestyle=":",
                    label=f"{kind} p90",
                )
            ax.set_xticks(
                range(len(ARRIVAL_BINS) - 1),
                ["in flight"]
                + [
                    f"<{end:g}" if math.isfinite(end) else f">{start:g}"
                    for start, end in zip(ARRIVAL_BINS[1:], ARRIVAL_BINS[2:])
                ],
                rotation=30,
            )
            lo = next(
                (
                    r["f_trust_lo"]
                    for r in summary
                    if r["system"] == system and r["iter"] == latest
                ),
                math.inf,
            )
            if math.isfinite(lo):
                ax.axhline(lo, color=MUTED, linestyle="--", linewidth=1)
                ax.annotate(
                    "trust_lo",
                    (len(ARRIVAL_BINS) - 2, lo),
                    xytext=(0, 3),
                    ha="right",
                    textcoords="offset points",
                    fontsize=8,
                    color=MUTED,
                )
            ax.set(
                title=f"System {system:03d}, iter {latest}: after impact",
                xlabel="time since arrival (ps)",
                ylabel="max_devi_f (eV/Å)",
            )
            ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out / "devi_arrival.png")
        plt.close(fig)


def plot_train(plt, rows, curves, out):
    if not rows:
        return
    iters = sorted({r["iter"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, quantity, unit in ((axes[0], "f", "eV/Å"), (axes[1], "e", "eV")):
        for i, split in enumerate(("trn", "val")):
            key = f"rmse_{quantity}_{split}"
            series = [
                [r[key] for r in rows if r["iter"] == it and r.get(key) is not None]
                for it in iters
            ]
            x = [it for it, s in zip(iters, series) if s]
            if not x:
                continue
            # Median: one bad model should show as a wide band, not move the line.
            mean = [float(np.median(s)) for s in series if s]
            ax.fill_between(
                x,
                [min(s) for s in series if s],
                [max(s) for s in series if s],
                color=SERIES[i],
                alpha=0.15,
                linewidth=0,
            )
            ax.plot(
                x,
                mean,
                marker="o",
                markersize=4,
                color=SERIES[i],
                label="training" if split == "trn" else "validation",
            )
        ax.set(
            title=f"Final {'force' if quantity == 'f' else 'energy'} RMSE (band: spread over models)",
            xlabel="iteration",
            ylabel=f"RMSE ({unit})",
        )
        integer_ticks(ax)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "train_rmse.png")
    plt.close(fig)

    latest = iters[-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, quantity, unit in ((axes[0], "f", "eV/Å"), (axes[1], "e", "eV")):
        for i, ((it, model), (names, data)) in enumerate(
            sorted(k for k in curves.items() if k[0][0] == latest)
        ):
            key = next(
                (
                    k
                    for k in (f"rmse_{quantity}_val", f"rmse_{quantity}_trn")
                    if k in names
                ),
                None,
            )
            if key is None:
                continue
            table = np.asarray(data)
            color = SERIES[i % len(SERIES)]
            # Raw rows are single batches: show them faintly, read the median.
            ax.plot(
                table[:, 0],
                table[:, names.index(key)],
                color=color,
                linewidth=0.6,
                alpha=0.25,
            )
            ax.plot(
                table[:, 0],
                smooth(table[:, names.index(key)], max(5, len(table) // 10)),
                color=color,
                linewidth=2,
                label=f"model {model}",
            )
        ax.set_yscale("log")
        ax.set(
            title=f"Iter {latest}: {'force' if quantity == 'f' else 'energy'} RMSE (rolling median)",
            xlabel="step",
            ylabel=f"RMSE ({unit})",
        )
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "train_lcurve.png")
    plt.close(fig)


def plot_fp(plt, summary, out):
    if not summary:
        return
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.array([r["iter"] for r in summary])
    bottom = np.zeros(len(x))
    for i, status in enumerate(("done", "stopped", "failed", "discarded", "other")):
        y = np.array([r[status] for r in summary], dtype=float)
        if not y.any():
            continue
        ax.bar(
            x,
            y,
            bottom=bottom,
            color=SERIES[i],
            edgecolor=SURFACE,
            linewidth=1,
            width=0.7,
            label=status,
        )
        bottom += y
    ax.set(title="FP tasks by outcome", xlabel="iteration", ylabel="tasks")
    integer_ticks(ax)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fp_status.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# entry point


def analyze(args):
    root = Path(args.JOB_DIR)
    jdata = read_json(root / args.param)
    if jdata is None:
        raise SystemExit(f"cannot read {root / args.param}")
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    iters = iterations(root, args.iters)
    if not iters:
        raise SystemExit(f"no iter.* folders selected in {root}")
    parts = PARTS if args.PART == "all" else (args.PART,)
    plt = None if args.no_plots else _pyplot()
    notes = []
    written = []

    if "parity" in parts:
        from dpgen.tools import parity

        machine = None
        if args.machine:
            machine = str(
                args.machine if Path(args.machine).exists() else root / args.machine
            )
        notes, written = parity.run(
            root,
            jdata,
            out,
            iters,
            choice=args.model,
            machine=machine,
            dp_command=args.dp_command,
            d3_command=parity.d3_command_from(machine, args.d3_command),
            plt=plt,
            style=(SERIES, INK, MUTED),
        )

    if "devi" in parts:
        frames = collect_frames(root, jdata, iters)
        summary = summarize_frames(frames)
        arrival = arrival_profile(frames)
        write_csv(out / "devi_summary.csv", summary)
        written.append("devi_summary.csv")
        if arrival:
            write_csv(out / "devi_arrival.csv", arrival)
            written.append("devi_arrival.csv")
        if args.frames:
            write_csv(out / "devi_frames.csv", frames)
            written.append("devi_frames.csv")
        notes += (
            devi_findings(summary) if summary else ["devi: no model_devi.out found"]
        )
        if plt:
            plot_devi(plt, frames, summary, arrival, out)
            written += ["devi_shares.png", "devi_cdf.png"] + (
                ["devi_arrival.png"] if arrival else []
            )

    if "train" in parts:
        rows, curves = collect_training(root, iters)
        if rows:
            columns = list(dict.fromkeys(k for r in rows for k in r))
            write_csv(out / "train_summary.csv", rows, columns)
            written.append("train_summary.csv")
        notes += train_findings(rows)
        if plt and rows:
            plot_train(plt, rows, curves, out)
            written += ["train_rmse.png", "train_lcurve.png"]

    if "fp" in parts:
        rows = collect_fp(root, iters)
        summary = summarize_fp(rows)
        if rows:
            write_csv(out / "fp_tasks.csv", rows)
            write_csv(out / "fp_summary.csv", summary)
            written += ["fp_tasks.csv", "fp_summary.csv"]
        notes += fp_findings(summary)
        if plt and summary:
            plot_fp(plt, summary, out)
            written.append("fp_status.png")

    text = [f"dpgen analyze {args.PART}: iterations {iters[0]}-{iters[-1]}", ""] + notes
    text += ["", f"files in {out}:"] + [f"  {name}" for name in written]
    if plt is None and not args.no_plots:
        text.append("  (no plots: matplotlib is not installed)")
    (out / "summary.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    return notes


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "analyze",
        help="Diagnose a run: exploration deviation, training convergence, FP losses.",
        description=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "PART",
        choices=(*PARTS, "parity", "all"),
        help="what to analyze; parity runs dp test and is not part of all",
    )
    parser.add_argument(
        "JOB_DIR", nargs="?", default=".", help="the DP-GEN job folder (default: .)"
    )
    parser.add_argument(
        "-p", "--param", default="param.json", help="param file inside JOB_DIR"
    )
    parser.add_argument(
        "-o", "--output", default="analysis", help="output folder inside JOB_DIR"
    )
    parser.add_argument(
        "-i",
        "--iters",
        help="iterations to include, e.g. 10-19 or 12,15 (default: all)",
    )
    parser.add_argument(
        "--frames",
        action="store_true",
        help="also write every explored frame to devi_frames.csv (large)",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="write tables and summary only"
    )
    parity = parser.add_argument_group("parity")
    parity.add_argument(
        "--model",
        default="0",
        help='parity: model index to test (default 0), or "mean" for the committee mean',
    )
    parity.add_argument(
        "--machine",
        help="parity: machine.json whose train section runs dp test as a job; "
        "without it dp test runs here",
    )
    parity.add_argument(
        "--dp-command",
        help="parity: DeePMD command, e.g. 'dp --pt-expt' (default: from the model "
        "suffix, or the machine train command)",
    )
    parity.add_argument(
        "--d3-command",
        help="parity: LAMMPS for stripping D3 from the test set when "
        "training_strip_d3 is set (default: machine d3_command, else lmp)",
    )
    parser.set_defaults(func=analyze)
    return parser
