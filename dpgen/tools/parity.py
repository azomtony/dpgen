"""Parity of model predictions against held-out FP single points.

``dpgen analyze parity`` merges every iteration's held-out labels
(``iter.*/02.fp/validation.SSS``, written for ``fp_holdout_fraction`` and for
plasma runs) into one fixed test set, then tests the models of each selected
iteration on it with ``dp test``:

- ``analysis/parity/testset/sys.SSS/<formula>``: the merged test set. If the
  run trains on D3-stripped labels (``training_strip_d3``), the test set is
  stripped the same way, so models and labels match.
- ``analysis/parity/runs/iter.N/<model>/...``: ``dp test`` outputs, reused
  while the test set and model are unchanged.
- ``analysis/parity.png``: one row per iteration, energy per atom and force
  components, all systems in each panel, RMSE in each panel.
- ``analysis/parity_summary.csv``: RMSE and MAE per iteration and system.
"""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np

SUFFIX_COMMANDS = {".pt2": "dp --pt-expt", ".pth": "dp --pt", ".pb": "dp"}
DETAIL_FILES = ("detail.e_peratom.out", "detail.f.out")
MAX_PLOTTED_FORCES = 200_000


# --------------------------------------------------------------------------
# test set


def holdout_dirs(root):
    """iter.*/02.fp/validation.SSS folders, as (iteration, system, path)."""
    found = []
    for path in sorted(Path(root).glob("iter.*/02.fp/validation.*")):
        iteration = path.parts[-3].split(".")[1]
        system = path.name.split(".")[1]
        if path.is_dir() and iteration.isdigit() and system.isdigit():
            found.append((int(iteration), int(system), path))
    return found


def _labeled_systems(path):
    import dpdata

    for type_raw in sorted(Path(path).rglob("type.raw")):
        system = dpdata.LabeledSystem(str(type_raw.parent), fmt="deepmd/npy")
        if len(system):
            yield system


def _digest_systems(systems):
    digest = hashlib.sha256()
    for system in systems:
        for name in sorted(system.data):
            value = system.data[name]
            digest.update(name.encode())
            if isinstance(value, np.ndarray):
                digest.update(str((value.shape, value.dtype)).encode())
                digest.update(np.ascontiguousarray(value).tobytes())
            else:
                digest.update(repr(value).encode())
    return digest.hexdigest()


def build_testset(root, out, jdata, d3_command="lmp", d3_runner=None):
    """Merge all held-out labels into ``out/testset``; strip D3 if training does.

    Returns (manifest, list of (system index, formula, path)).
    """
    import dpdata

    from dpgen.generator.lib import d3

    params = d3.strip_options(jdata)
    merged, sources = {}, {}
    for iteration, system, path in holdout_dirs(root):
        for labeled in _labeled_systems(path):
            if system not in merged:
                merged[system] = dpdata.MultiSystems(type_map=jdata["type_map"])
            merged[system].append(labeled)
            sources.setdefault(iteration, 0)
            sources[iteration] += len(labeled)
    testset = Path(out) / "testset"
    manifest_path = testset / "testset.json"
    ordered = [
        (system, formula, merged[system][formula])
        for system in sorted(merged)
        for formula in sorted(merged[system].systems)
    ]
    digest = _digest_systems(item[2] for item in ordered)
    manifest = {
        "digest": digest,
        "d3": params,
        "frames_by_source_iteration": {str(k): v for k, v in sorted(sources.items())},
        "systems": [
            {"system": system, "formula": formula, "frames": len(labeled)}
            for system, formula, labeled in ordered
        ],
    }
    entries = [
        (system, formula, testset / f"sys.{system:03d}" / formula)
        for system, formula, _ in ordered
    ]
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("digest") == digest and previous.get("d3") == params:
            return manifest, entries
    if testset.exists():
        shutil.rmtree(testset)
    for (system, formula, labeled), (_, _, path) in zip(ordered, entries):
        if params is not None:
            labeled = labeled.copy()
            energies, forces, virials = d3.compute_d3(
                labeled, params, d3_command, d3_runner or d3._run_lammps
            )
            labeled.data["energies"] = labeled["energies"] - energies
            labeled.data["forces"] = labeled["forces"] - forces
            if "virials" in labeled.data:
                labeled.data["virials"] = labeled["virials"] - virials
        labeled.to_deepmd_npy(str(path))
    testset.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest, entries


# --------------------------------------------------------------------------
# models and dp test


def iteration_models(root, iteration):
    train = Path(root) / f"iter.{iteration:06d}" / "00.train"
    return sorted(
        path
        for path in train.glob("graph.[0-9][0-9][0-9].*")
        if path.suffix in SUFFIX_COMMANDS
    )


def pick_models(models, choice):
    """``choice`` is a model index ("0") or "mean" for the committee mean."""
    if choice == "mean":
        return models
    index = int(choice)
    return [models[index]] if index < len(models) else []


def _model_stamp(model):
    stat = Path(model).stat()
    return [str(Path(model).resolve()), stat.st_size, int(stat.st_mtime)]


def _done(case, digest):
    stamp = case["run_dir"] / "parity.json"
    if not stamp.exists() or not all(
        (case["run_dir"] / name).exists() for name in DETAIL_FILES
    ):
        return False
    previous = json.loads(stamp.read_text())
    return previous == {"testset": digest, "model": _model_stamp(case["model"])}


def make_cases(root, iterations, choice, entries, runs):
    cases, skipped = [], []
    for iteration in iterations:
        models = pick_models(iteration_models(root, iteration), choice)
        if not models:
            skipped.append(iteration)
            continue
        for model in models:
            for system, formula, path in entries:
                run_task = (
                    f"iter.{iteration:06d}/{model.stem}/sys.{system:03d}/{formula}"
                )
                cases.append(
                    {
                        "iteration": iteration,
                        "system_index": system,
                        "system": path,
                        "system_name": f"sys.{system:03d}/{formula}",
                        "model": model,
                        "run_task": run_task,
                        "run_dir": Path(runs) / run_task,
                    }
                )
    return cases, skipped


def run_dp_test(cases, runs, digest, dp_command=None, machine=None):
    """Run ``dp test`` for every case not already done for this test set."""
    todo = [case for case in cases if not _done(case, digest)]
    if todo and machine is not None:
        from dpgen.tools.test_model import submit_model_tests

        submit_model_tests(
            todo, runs, machine, "train", dp_command=dp_command, keep_going=True
        )
    elif todo:
        for case in todo:
            case["run_dir"].mkdir(parents=True, exist_ok=True)
            command = dp_command or SUFFIX_COMMANDS[case["model"].suffix]
            result = subprocess.run(
                [
                    *shlex.split(command),
                    "test",
                    "-m",
                    str(case["model"]),
                    "-s",
                    str(case["system"]),
                    "-n",
                    "0",
                    "-d",
                    str(case["run_dir"] / "detail"),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            (case["run_dir"] / "dp_test.log").write_text(result.stdout)
    failed = []
    for case in todo:
        if all((case["run_dir"] / name).exists() for name in DETAIL_FILES):
            (case["run_dir"] / "parity.json").write_text(
                json.dumps({"testset": digest, "model": _model_stamp(case["model"])})
            )
        else:
            failed.append(case)
    return failed


def read_detail(run_dir):
    energy = np.atleast_2d(np.loadtxt(Path(run_dir) / "detail.e_peratom.out"))
    force = np.atleast_2d(np.loadtxt(Path(run_dir) / "detail.f.out"))
    return energy[:, 0], energy[:, 1], force[:, :3].ravel(), force[:, 3:6].ravel()


def gather(cases):
    """{iteration: {system index: [e_data, e_pred, f_data, f_pred]}}.

    With several models per iteration (committee mean), predictions are
    averaged per frame and component.
    """
    grouped = {}
    for case in cases:
        if not all((case["run_dir"] / n).exists() for n in DETAIL_FILES):
            continue
        key = (case["iteration"], case["system_index"], case["system_name"])
        grouped.setdefault(key, []).append(read_detail(case["run_dir"]))
    results = {}
    for (iteration, system, _), runs in sorted(grouped.items()):
        e_data, _, f_data, _ = runs[0]
        e_pred = np.mean([run[1] for run in runs], axis=0)
        f_pred = np.mean([run[3] for run in runs], axis=0)
        slot = results.setdefault(iteration, {}).setdefault(system, [[], [], [], []])
        for column, values in zip(slot, (e_data, e_pred, f_data, f_pred)):
            column.append(values)
    return {
        iteration: {
            system: [np.concatenate(column) for column in columns]
            for system, columns in systems.items()
        }
        for iteration, systems in results.items()
    }


def errors(data, pred):
    diff = pred - data
    return float(np.sqrt(np.mean(diff**2)) * 1000), float(np.mean(np.abs(diff)) * 1000)


# --------------------------------------------------------------------------
# report


def system_labels(jdata):
    """Short names from sys_configs, e.g. 'arch/*/*/POSCAR' -> 'arch'."""
    labels = {}
    for index, entry in enumerate(jdata.get("sys_configs", [])):
        pattern = entry[0] if isinstance(entry, list) and entry else str(entry)
        parts = [p for p in Path(pattern).parts if not any(c in p for c in "*?[")]
        parts = [p for p in parts if p != "POSCAR"]
        labels[index] = parts[0] if parts else f"sys.{index:03d}"
    return labels


def summary_rows(results, choice, labels):
    rows = []
    for iteration, systems in sorted(results.items()):
        model = "committee mean" if choice == "mean" else f"graph.{int(choice):03d}"
        blocks = [("all", list(systems.values()))] + [
            (labels.get(s, f"sys.{s:03d}"), [v]) for s, v in sorted(systems.items())
        ]
        for name, values in blocks:
            e_data = np.concatenate([v[0] for v in values])
            e_pred = np.concatenate([v[1] for v in values])
            f_data = np.concatenate([v[2] for v in values])
            f_pred = np.concatenate([v[3] for v in values])
            e_rmse, e_mae = errors(e_data, e_pred)
            f_rmse, f_mae = errors(f_data, f_pred)
            rows.append(
                {
                    "iter": iteration,
                    "model": model,
                    "system": name,
                    "frames": len(e_data),
                    "energy_rmse_meV_atom": round(e_rmse, 4),
                    "energy_mae_meV_atom": round(e_mae, 4),
                    "force_rmse_meV_A": round(f_rmse, 4),
                    "force_mae_meV_A": round(f_mae, 4),
                }
            )
    return rows


def findings(rows, manifest):
    totals = [r for r in rows if r["system"] == "all"]
    sources = manifest["frames_by_source_iteration"]
    notes = [
        f"parity: test set {sum(sources.values())} held-out frames from "
        f"{len(sources)} iteration(s)"
        + (", D3 removed to match training" if manifest["d3"] else "")
    ]
    for row in totals:
        notes.append(
            f"parity iter {row['iter']} ({row['model']}): energy RMSE "
            f"{row['energy_rmse_meV_atom']:.2f} meV/atom, force RMSE "
            f"{row['force_rmse_meV_A']:.1f} meV/Å"
        )
    if len(totals) >= 2:
        first, last = totals[0], totals[-1]
        notes.append(
            f"parity: force RMSE {first['force_rmse_meV_A']:.1f} -> "
            f"{last['force_rmse_meV_A']:.1f} meV/Å from iter {first['iter']} "
            f"to {last['iter']} on the same test set"
        )
    return notes


def _thin(values, limit, rng):
    if len(values[0]) <= limit:
        return values
    keep = np.sort(rng.choice(len(values[0]), limit, replace=False))
    return [v[keep] for v in values]


def plot_parity(plt, results, rows, choice, labels, out, palette, ink, muted):
    """One row per iteration: energy per atom | force components."""
    iterations = sorted(results)
    systems = sorted({s for its in results.values() for s in its})
    named = systems[: len(palette)]
    color = {s: palette[i] for i, s in enumerate(named)}
    other = muted  # systems beyond the palette fold into one "Other" series
    fig, axes = plt.subplots(
        len(iterations), 2, figsize=(9, 3.9 * len(iterations)), squeeze=False
    )
    totals = {r["iter"]: r for r in rows if r["system"] == "all"}
    rng = np.random.default_rng(0)
    for row, iteration in enumerate(iterations):
        for column, (index, unit, title) in enumerate(
            ((0, "meV/atom", "Energy (eV/atom)"), (2, "meV/Å", "Force (eV/Å)"))
        ):
            ax = axes[row][column]
            lows, highs = [], []
            present = sorted(results[iteration])
            total_forces = sum(len(results[iteration][s][2]) for s in present)
            for system in present:
                data, pred = results[iteration][system][index : index + 2]
                if index == 2:
                    share = max(1, MAX_PLOTTED_FORCES * len(data) // total_forces)
                    data, pred = _thin([data, pred], share, rng)
                dense = len(data) > 5000
                ax.scatter(
                    data,
                    pred,
                    s=2 if dense else 16,
                    color=color.get(system, other),
                    alpha=0.3 if dense else 0.8,
                    linewidths=0,
                    rasterized=True,
                    zorder=2,
                )
                lows.append(min(data.min(), pred.min()))
                highs.append(max(data.max(), pred.max()))
            low, high = min(lows), max(highs)
            pad = 0.03 * (high - low or 1)
            ax.plot(
                [low - pad, high + pad],
                [low - pad, high + pad],
                color=muted,
                linewidth=1,
                linestyle="--",
                zorder=1,
            )
            ax.set_xlim(low - pad, high + pad)
            ax.set_ylim(low - pad, high + pad)
            ax.set_aspect("equal")
            ax.set_xlabel(f"DFT {title}")
            ax.set_ylabel(f"Model {title}")
            total = totals[iteration]
            key = "energy" if index == 0 else "force"
            suffix = "meV_atom" if index == 0 else "meV_A"
            ax.text(
                0.04,
                0.96,
                f"RMSE {total[f'{key}_rmse_{suffix}']:.3g} {unit}\n"
                f"MAE {total[f'{key}_mae_{suffix}']:.3g} {unit}\n"
                f"{total['frames']} frames",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color=ink,
            )
            ax.set_title(f"iter {iteration} · {total['model']}", loc="left")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=color[s],
            label=labels.get(s, f"sys.{s:03d}"),
        )
        for s in named
    ]
    if len(systems) > len(named):
        handles.append(
            plt.Line2D([], [], marker="o", linestyle="", color=other, label="Other")
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(handles), 6),
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (3.9 * len(iterations))))
    fig.savefig(Path(out) / "parity.png")
    plt.close(fig)


def run(
    root,
    jdata,
    out,
    iterations,
    choice="0",
    machine=None,
    dp_command=None,
    d3_command="lmp",
    plt=None,
    style=None,
    d3_runner=None,
):
    """Build the test set, test the selected iterations, write the report.

    Returns (notes, written files).
    """
    from dpgen.tools.analyze import write_csv

    if choice != "mean" and not str(choice).isdigit():
        raise SystemExit('--model must be a model index (e.g. 0) or "mean"')
    work = Path(out) / "parity"
    if not holdout_dirs(root):
        return [
            "parity: no held-out labels (iter.*/02.fp/validation.*) yet; set "
            "fp_holdout_fraction in param.json so new iterations keep some"
        ], []
    manifest, entries = build_testset(root, work, jdata, d3_command, d3_runner)
    runs = work / "runs"
    cases, skipped = make_cases(root, iterations, str(choice), entries, runs)
    notes = [
        f"parity: iter {it} has no trained model {choice}, skipped" for it in skipped
    ]
    if not cases:
        return notes + ["parity: no iteration with trained models selected"], []
    failed = run_dp_test(cases, runs, manifest["digest"], dp_command, machine)
    notes += [
        f"parity: dp test failed for {case['run_task']}; see "
        f"{case['run_dir'] / 'dp_test.log'}"
        for case in failed
    ]
    results = gather(cases)
    if not results:
        return notes + ["parity: no dp test results to report"], []
    labels = system_labels(jdata)
    rows = summary_rows(results, str(choice), labels)
    write_csv(Path(out) / "parity_summary.csv", rows)
    written = ["parity_summary.csv"]
    notes = findings(rows, manifest) + notes
    if plt is not None:
        plot_parity(plt, results, rows, str(choice), labels, out, *style)
        written.append("parity.png")
    return notes, written


def d3_command_from(machine_file, explicit):
    if explicit:
        return explicit
    if machine_file and os.path.exists(machine_file):
        from dpgen.tools.test_model import _load_machine_file

        return _load_machine_file(machine_file).get("d3_command", "lmp")
    return "lmp"
