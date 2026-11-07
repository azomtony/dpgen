"""Plasma generation, variable composition, event sampling and crash recovery."""

import copy
import csv
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dpdata
import numpy as np

from dpgen.generator.lib import plasma
from dpgen.generator.lib import plasma_runtime as worker
from dpgen.generator.run import (
    _abspath_fp_incar,
    _make_fp_vasp_inner,
    _select_vasp_incar_path,
    make_fp_vasp_incar,
    make_model_devi,
    make_train,
    post_fp_check_fail,
    post_fp_vasp,
    run_md_model_devi,
    sys_link_fp_vasp_pp,
)


def snapshot(step, time, oxygen=0):
    atoms = [f"{i + 1} 1 {5 + i} 5 {5 + i} 0.1 0.2 0.3 0 0 0" for i in range(4)]
    atoms += [f"{i + 5} 2 10 10 {12 + i} 1 2 -3 0 0 0" for i in range(oxygen)]
    return (
        f"ITEM: TIME\n{time:.17g}\nITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(atoms)}\n"
        "ITEM: BOX BOUNDS pp pp ff\n0 30\n0 30\n0 50\n"
        "ITEM: ATOMS id type x y z vx vy vz fx fy fz\n" + "\n".join(atoms) + "\n"
    )


class FakeLammps:
    """Exercise the real disk transaction protocol using bounded physical-time runs."""

    def __init__(self, fail_call=None, force=0.2):
        self.calls = 0
        self.fail_call = fail_call
        self.force = force
        self.inputs = []

    def __call__(self, command, check):
        self.calls += 1
        self.assert_check = check
        path = Path(command[-1])
        text = path.read_text()
        self.inputs.append(text)
        directory = path.parent
        match = re.search(r'read_restart "(.+?)/restart.bin"', text)
        prior = (
            json.loads(Path(match[1], "restart.bin").read_text())
            if match
            else {"step": 0, "time": 0.0, "oxygen": 0, "arrivals": 0}
        )
        current = dict(prior)
        if "fix addO " in text:
            current["oxygen"] += 1
            current["arrivals"] += 1
            current["step"] += 1
            current["time"] += 0.01
        if "fix addO " in text:
            (directory / "arrival.txt").write_text(str(current["time"]))
        target = float(
            re.search(r"fix block_end all halt 1 v_clock >= ([^ ]+)", text).group(1)
        )
        steps = int(
            re.search(r"variable checkpoint_end internal \$\(step\+(\d+)", text).group(
                1
            )
        )
        frequency = int(re.search(r"out_freq (\d+)", text).group(1))
        rows = []
        for _ in range(steps):
            current["step"] += 1
            current["time"] += 0.05
            if current["step"] % frequency == 0:
                rows.append([current["step"], 0, 0, 0, self.force, 0, 0])
                (directory / "traj" / f"{current['step']}.lammpstrj").write_text(
                    snapshot(current["step"], current["time"], current["oxygen"])
                )
            if current["time"] >= target:
                break
        (directory / "model_devi.out").write_text(
            "# deviation\n" + "".join(" ".join(map(str, row)) + "\n" for row in rows)
        )
        (directory / "restart.bin").write_text(json.dumps(current))
        (directory / "progress.txt").write_text(
            f"{current['step']} {current['time']:.17g} 0.05 {current['oxygen']}"
        )
        if self.calls == self.fail_call:
            raise RuntimeError(
                "simulated preemption after insertion/output, before commit"
            )


class TestPlasma(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.original = os.getcwd()
        os.chdir(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(os.chdir, self.original)
        Path("slab.dump").write_text(snapshot(0, 0))
        dpdata.System(
            "slab.dump", fmt="lammps/dump", type_map=["Cu", "O"]
        ).to_vasp_poscar("POSCAR")
        self.job = {
            "sys_idx": [0],
            "ensemble": "plasma",
            "temps": [293.0],
            "trj_freq": 1,
            "plasma_stop_factor": 2.0,
            "plasma": {
                "nions": 2,
                "ratio": 1,
                "eq_time": 0.2,
                "eq_trj_freq": 2,
                "ion_dt": 0.4,
                "checkpoint_freq": 3,
                "seed": 123,
            },
            "selection": {"validation_task_max": 0, "min_time_separation": 0.01},
        }
        self.data = {
            "type_map": ["Cu", "O"],
            "mass_map": [63.546, 15.999],
            "sys_configs": [[str(Path("POSCAR").resolve())]],
            "model_devi_dt": 0.001,
            "model_devi_skip": 0,
            "model_devi_f_trust_lo": 0.1,
            "model_devi_f_trust_hi": 0.3,
            "model_devi_jobs": [copy.deepcopy(self.job) for _ in range(3)],
            "fp_style": "vasp",
            "model_devi_restart_from_last": True,
        }
        self.machine = {"deepmd_version": "3.2"}

    def make(self, iteration=0):
        path = Path(f"iter.{iteration:06d}/00.train")
        path.mkdir(parents=True)
        for index in range(2):
            (path / f"graph.{index:03d}.pb").touch()
        self.assertTrue(make_model_devi(iteration, self.data, self.machine))
        return Path(f"iter.{iteration:06d}/01.model_devi/task.000.000000")

    def run_task(self, task, fake=None):
        original = os.getcwd()
        os.chdir(task)
        try:
            return worker.run("lmp", execute=fake or FakeLammps())
        finally:
            os.chdir(original)

    def test_generated_committee_and_submission(self):
        task = self.make()
        text = (task / "input.lammps").read_text()
        self.assertIn("graph.000.pb ../graph.001.pb out_freq 2", text)
        self.assertIn("boundary p p f", text)
        self.assertIn("fix heat mobile langevin", text)
        self.assertIn("time yes", text)
        machine = dict(
            self.machine,
            model_devi_command="lmp -sf kk",
            model_devi_machine={},
            model_devi_resources={},
            model_devi_group_size=1,
            api_version="1.0",
        )
        with patch("dpgen.dispatcher.Dispatcher.make_submission") as submit:
            run_md_model_devi(0, self.data, machine)
            call = submit.call_args.kwargs
            self.assertIn("plasma_runtime.py", call["commands"][0])
            self.assertIn("lmp -sf kk", call["commands"][0])
            self.assertIn("checkpoints", call["backward_files"])

    def test_preemption_preserves_dose_and_absolute_target(self):
        task = self.make()
        fake = FakeLammps(fail_call=3)
        with self.assertRaisesRegex(RuntimeError, "preemption"):
            self.run_task(task, fake)
        resumed = self.run_task(task)
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(resumed["arrivals"], 4)
        self.assertAlmostEqual(resumed["final_time"], 1.0)
        self.assertGreaterEqual(resumed["time"], 1.0)
        self.assertLess(resumed["time"], 1.06)
        events = json.loads((task / "deposition.json").read_text())
        self.assertEqual([event["event"] for event in events], [1, 2, 3, 4])
        self.assertEqual(
            [event["kind"] for event in events], ["ion", "neutral", "ion", "neutral"]
        )
        # Completed jobs are idempotent, not a new dose.
        fake = FakeLammps()
        self.run_task(task, fake)
        self.assertEqual(fake.calls, 0)
        # A corrupt newest checkpoint falls back and still finishes the same dose.
        checkpoints = sorted((task / "checkpoints").glob("*/state.json"))
        (checkpoints[-1].parent / "restart.bin").write_bytes(b"damaged")
        recovered = self.run_task(task)
        self.assertEqual(recovered["arrivals"], 4)
        self.assertEqual(len(json.loads((task / "deposition.json").read_text())), 4)

    def test_preemption_during_equilibration_keeps_original_target(self):
        task = self.make()
        with self.assertRaisesRegex(RuntimeError, "preemption"):
            self.run_task(task, FakeLammps(fail_call=2))
        result = self.run_task(task)
        self.assertAlmostEqual(result["eq_target"], 0.2)
        self.assertAlmostEqual(result["final_time"], 1.0)
        self.assertEqual(result["arrivals"], 4)

    def test_new_segment_keeps_velocities_and_skips_equilibration(self):
        task = self.make()
        self.run_task(task)
        next_task = self.make(1)
        config = json.loads((next_task / "plasma.json").read_text())
        self.assertTrue(config["eq_complete"])
        self.assertEqual(config["nions"], 2)
        self.assertNotEqual(
            config["seed"], json.loads((task / "plasma.json").read_text())["seed"]
        )
        self.assertIn("Velocities", (next_task / "conf.lmp").read_text())
        self.assertIn(
            "fix heat bath langevin", (next_task / "input.lammps").read_text()
        )
        result = self.run_task(next_task)
        self.assertAlmostEqual(result["final_time"], 0.8)
        self.assertEqual(result["arrivals"], 4)

    def test_hard_stop_is_distinct_from_failed_frames(self):
        task = self.make()
        result = self.run_task(task, FakeLammps(force=0.4))
        self.assertEqual(result["status"], "complete")
        other = self.make(1)  # All frames failed: original slab fallback.
        result = self.run_task(other, FakeLammps(force=0.61))
        self.assertEqual(result["status"], "trust_stop")
        self.assertIn("stop_frame", result)
        fake = FakeLammps()
        self.run_task(other, fake)
        self.assertEqual(fake.calls, 0)

    def test_event_budgets_ignore_quiet_frame_accuracy(self):
        task = self.make()
        self.run_task(task)
        job = self.data["model_devi_jobs"][0]
        job["selection"] = {
            "eq_task_max": 0,
            "ion_task_max": 2,
            "neutral_task_max": 2,
            "validation_task_max": 2,
            "max_frames_per_event": 1,
            "min_time_separation": 0.01,
        }
        accurate, selected, failed, counts, metadata = plasma.select_frames(
            [str(task)], job, self.data, 0, 4
        )
        self.assertEqual(len(failed), 0)
        training = [ref for ref in selected if not metadata[tuple(ref)]["validation"]]
        validation = [ref for ref in selected if metadata[tuple(ref)]["validation"]]
        self.assertEqual(len(validation), 2)
        self.assertTrue(training)
        self.assertTrue(all(metadata[tuple(ref)]["kind"] != "eq" for ref in selected))
        events = [(ref[0], metadata[tuple(ref)]["event"]) for ref in training]
        self.assertEqual(len(events), len(set(events)))
        self.assertFalse(set(map(tuple, training)) & set(map(tuple, validation)))
        training_events = {(ref[0], metadata[tuple(ref)]["event"]) for ref in training}
        validation_events = {
            (ref[0], metadata[tuple(ref)]["event"]) for ref in validation
        }
        self.assertFalse(training_events & validation_events)
        # Existing FP creation must honor plasma budgets even at 100% accurate
        # and below ordinary fp_task_min, and preserve exact changing counts.
        self.data.update(fp_accurate_threshold=0.0, fp_task_min=100, fp_task_max=4)
        output = Path("iter.000000/02.fp")
        output.mkdir()
        _make_fp_vasp_inner(
            0,
            str(task.parent),
            str(output),
            0,
            1e10,
            1e10,
            0.1,
            0.3,
            100,
            4,
            [],
            ["Cu", "O"],
            self.data,
        )
        labels = sorted(output.glob("task.*.*"))
        self.assertEqual(len(labels), len(selected))
        oxygen_counts = set()
        for label in labels:
            structure = dpdata.System(
                str(label / "POSCAR"), fmt="vasp/poscar", type_map=["Cu", "O"]
            )
            oxygen_counts.add(structure.data["atom_numbs"][1])
            self.assertIn("plasma_frame", json.loads((label / "job.json").read_text()))
        self.assertGreater(len(oxygen_counts), 1)

    def test_selection_report_tracks_uncertainty_and_fp_origins(self):
        from dpgen import dlog

        task = self.make()
        self.run_task(task)
        frames = json.loads((task / "frames.json").read_text())
        rows = np.atleast_2d(np.loadtxt(task / "model_devi.out"))
        for row in rows:
            kind = frames[str(int(row[0]))]["kind"]
            row[4] = {"eq": 0.05, "ion": 0.4, "neutral": 0.2}[kind]
        np.savetxt(task / "model_devi.out", rows)
        output = Path("iter.000000/02.fp")
        output.mkdir()
        with self.assertLogs(dlog.name, level="INFO") as captured:
            _make_fp_vasp_inner(
                0,
                str(task.parent),
                str(output),
                0,
                1e10,
                1e10,
                0.1,
                0.3,
                0,
                4,
                [],
                ["Cu", "O"],
                self.data,
            )
        report = json.loads((output / "plasma-selection.000.json").read_text())
        self.assertEqual(report["phases"]["eq"]["candidate_rate"], 0)
        self.assertEqual(report["phases"]["ion"]["failed_rate"], 1)
        self.assertEqual(report["phases"]["neutral"]["candidate_rate"], 1)
        self.assertEqual(report["phases"]["neutral"]["fraction_of_candidates"], 1)
        self.assertEqual(report["phases"]["neutral"]["fraction_of_fp"], 1)
        labels = sorted(output.glob("task.*.*"))
        self.assertEqual(report["totals"]["fp_created"], len(labels))
        self.assertEqual(
            sum(phase["sampled"] for phase in report["phases"].values()), len(rows)
        )
        self.assertEqual(sum(event["sampled"] for event in report["events"]), len(rows))
        with (output / "plasma-origins.000.csv").open() as stream:
            origins = list(csv.DictReader(stream))
        self.assertEqual(len(origins), len(labels))
        self.assertEqual(
            len([line for line in captured.output if "Plasma FP " in line]), len(labels)
        )
        for origin in origins:
            self.assertEqual(origin["kind"], "neutral")
            self.assertEqual(origin["source_task"], str(task))
            saved = json.loads((Path(origin["fp_task"]) / "job.json").read_text())[
                "plasma_frame"
            ]
            self.assertEqual(int(origin["step"]), saved["step"])
            self.assertEqual(float(origin["time_ps"]), saved["time"])
        text = (output / "plasma-selection.000.log").read_text()
        self.assertIn("FP origins:", text)
        self.assertIn("candidate_%", text)
        self.assertIn("not spatial regions", text)

    def test_selection_report_distinguishes_selected_from_geometry_skips(self):
        task = self.make()
        self.run_task(task)
        output = Path("iter.000000/02.fp")
        output.mkdir()
        self.data["fp_skip_bad_box"] = "length_ratio:3"
        with patch("dpgen.generator.run.check_bad_box", return_value=True):
            _make_fp_vasp_inner(
                0,
                str(task.parent),
                str(output),
                0,
                1e10,
                1e10,
                0.1,
                0.3,
                0,
                4,
                [],
                ["Cu", "O"],
                self.data,
            )
        report = json.loads((output / "plasma-selection.000.json").read_text())
        self.assertGreater(report["totals"]["training_selected"], 0)
        self.assertEqual(report["totals"]["training_created"], 0)
        self.assertEqual(report["totals"]["fp_created"], 0)
        self.assertTrue(
            all(phase["fraction_of_fp"] == 0 for phase in report["phases"].values())
        )

    def test_holdout_labels_are_not_in_training_data(self):
        task = self.make()
        self.run_task(task)
        self.data["model_devi_jobs"][0]["selection"]["validation_task_max"] = 2
        output = Path("iter.000000/02.fp")
        output.mkdir()
        _make_fp_vasp_inner(
            0,
            str(task.parent),
            str(output),
            0,
            1e10,
            1e10,
            0.1,
            0.3,
            0,
            3,
            [],
            ["Cu", "O"],
            self.data,
        )
        labels = sorted(output.glob("task.*.*"))
        Path("pseudopotentials").mkdir()
        Path("pseudopotentials/Cu").write_text("Cu\n")
        Path("pseudopotentials/O").write_text("O\n")
        self.data.update(fp_pp_path="pseudopotentials", fp_pp_files=["Cu", "O"])
        sys_link_fp_vasp_pp(0, self.data)
        species_sets = set()
        for label in labels:
            species = dpdata.System(str(label / "POSCAR"), fmt="vasp/poscar").data[
                "atom_names"
            ]
            species_sets.add(tuple(species))
            self.assertEqual(
                (label / "POTCAR").read_text(), "".join(name + "\n" for name in species)
            )
        self.assertEqual(species_sets, {("Cu",), ("Cu", "O")})
        for label in labels:
            (label / "OUTCAR").touch()
        # Patch only the external VASP parser; exercise real MultiSystems output.
        original = dpdata.LabeledSystem

        def parse(filename=None, **kwargs):
            if filename and str(filename).endswith("OUTCAR"):
                structure = dpdata.System(
                    str(Path(filename).parent / "POSCAR"),
                    fmt="vasp/poscar",
                    type_map=["Cu", "O"],
                )
                data = copy.deepcopy(structure.data)
                data.update(
                    energies=np.array(
                        [float(Path(filename).parent.name.split(".")[-1])]
                    ),
                    forces=np.zeros_like(data["coords"]),
                    virials=np.zeros((1, 3, 3)),
                )
                return original(data=data)
            return original(filename, **kwargs)

        with patch("dpgen.generator.run.dpdata.LabeledSystem", side_effect=parse):
            post_fp_vasp(0, self.data)
        train = dpdata.MultiSystems().from_deepmd_npy(str(output / "data.000"))
        validation = dpdata.MultiSystems().from_deepmd_npy(
            str(output / "validation.000")
        )
        train_energy = {
            float(value) for system in train for value in system.data["energies"]
        }
        val_energy = {
            float(value) for system in validation for value in system.data["energies"]
        }
        self.assertEqual(len(val_energy), 2)
        self.assertFalse(train_energy & val_energy)
        post_fp_check_fail(0, self.data)
        # Exercise real next-iteration data discovery: every composition trains,
        # while the separately stored validation groups remain excluded.
        fixtures = Path(__file__).parent / "generator"
        training = json.loads((fixtures / "param-mg-vasp.json").read_text())
        training.update(self.data)
        training.update(
            init_data_prefix=str(Path("initial-data").resolve()),
            init_data_sys=["system"],
            init_batch_size=[1],
            sys_batch_size=["auto"],
            numb_models=2,
            fp_task_min=0,
        )
        source = next(iter(train))
        source.to_deepmd_npy("initial-data/system")
        make_train(1, training, self.machine, force_train=True)
        generated = json.loads(Path("iter.000001/00.train/000/input.json").read_text())
        paths = generated["training"]["training_data"]["systems"]
        self.assertFalse(any("validation." in path for path in paths))
        for group in (output / "data.000").iterdir():
            if group.is_dir():
                self.assertTrue(
                    any(path.endswith("data.000/" + group.name) for path in paths)
                )

    def test_monitor_checks_every_sample_and_handles_nonfinite(self):
        task = self.make()
        config = json.loads((task / "plasma.json").read_text())
        folder = task / "monitor-test"
        folder.mkdir()
        for force, expected in ((0.4, False), (0.6, True), (float("nan"), True)):
            (folder / "model_devi.out").write_text(f"10 0 0 0 {force} 0 0\n")
            self.assertEqual(worker.monitor(folder, task / "plasma.json"), expected)
            self.assertIn(str(int(expected)), (folder / "control.lammps").read_text())
        text = (task / "input.lammps").read_text()
        self.assertIn("label plasma_sample", text)
        self.assertIn("--monitor", text)
        self.assertEqual(config["stop_factor"], 2)

    def test_no_valid_checkpoint_refuses_to_repeat_dose(self):
        task = self.make()
        self.run_task(task)
        for path in (task / "checkpoints").glob("*/restart.bin"):
            path.write_bytes(b"invalid")
        with self.assertRaisesRegex(RuntimeError, "refusing to restart"):
            self.run_task(task)

    def test_continuation_uses_matching_energy_and_falls_back_backward(self):
        task = self.make()
        self.run_task(task)
        rows = np.atleast_2d(np.loadtxt(task / "model_devi.out"))
        rows[-1, 4] = 0.4
        np.savetxt(task / "model_devi.out", rows)
        previous_step = int(rows[-2, 0])
        next_task = self.make(1)
        self.assertEqual(
            json.loads((next_task / "job.json").read_text())["restart"]["step"],
            previous_step,
        )
        self.run_task(next_task)
        self.data["model_devi_jobs"][2]["plasma"]["eion"] = 20
        fresh = self.make(2)
        config = json.loads((fresh / "plasma.json").read_text())
        self.assertFalse(config.get("eq_complete", False))
        self.assertEqual(
            json.loads((fresh / "job.json").read_text())["restart"]["source"], "initial"
        )

    def test_validation_and_sweeps(self):
        for key, value in (
            ("nions", 0),
            ("ratio", -1),
            ("surface", 123),
            ("eq_time", float("nan")),
        ):
            job = copy.deepcopy(self.job)
            job["plasma"][key] = value
            with self.assertRaises(ValueError):
                list(plasma.settings(job))
        self.data["model_devi_jobs"][0]["plasma"]["eion"] = [5, 10]
        self.make()
        tasks = sorted(Path("iter.000000/01.model_devi").glob("task.*.*"))
        self.assertEqual(len(tasks), 2)
        self.assertEqual(
            [json.loads((task / "plasma.json").read_text())["eion"] for task in tasks],
            [5, 10],
        )

    def test_surfaces_follow_system_order_without_cross_product(self):
        source = self.data["sys_configs"][0]
        self.data["sys_configs"] = [source for _ in range(22)]
        job = self.data["model_devi_jobs"][0]
        job["sys_idx"] = [21, 19, 20]
        job["plasma"].update(surface=[111, 100, 110], eion=[5, 10])
        self.make()
        tasks = sorted(Path("iter.000000/01.model_devi").glob("task.*.*"))
        self.assertEqual(len(tasks), 6)
        for task in tasks:
            system = int(task.name.split(".")[1])
            config = json.loads((task / "plasma.json").read_text())
            self.assertEqual(config["surface"], {19: 100, 20: 110, 21: 111}[system])
            self.assertIn(config["eion"], [5, 10])
        # The original input list must remain intact across task generation.
        self.assertEqual(job["plasma"]["surface"], [111, 100, 110])

    def test_surface_scalar_broadcast_and_invalid_lists(self):
        job = copy.deepcopy(self.job)
        job["sys_idx"] = [19, 20, 21]
        self.assertEqual(
            [next(plasma.settings(job, i))["surface"] for i in range(3)],
            [100, 100, 100],
        )
        for surfaces in ([100, 110], [], [100, 110, 123]):
            with self.subTest(surfaces=surfaces):
                job["plasma"]["surface"] = surfaces
                with self.assertRaises(ValueError):
                    [list(plasma.settings(job, i)) for i in range(3)]

    def test_fractional_budgets_scale_and_round_to_total(self):
        job = copy.deepcopy(self.job)
        job["selection"] = {
            "eq_fraction": 0.1,
            "ion_fraction": 0.3,
            "neutral_fraction": 0.5,
            "validation_fraction": 0.1,
        }
        for total, expected in (
            (10, [1, 3, 5, 1]),
            (30, [3, 9, 15, 3]),
            (100, [10, 30, 50, 10]),
        ):
            options, budgets = plasma.selection_budgets(job, total)
            self.assertFalse(options["_legacy"])
            self.assertEqual(list(budgets.values()), expected)
        for total in range(11):
            _, budgets = plasma.selection_budgets(job, total)
            self.assertEqual(sum(budgets.values()), total)
        for invalid in (
            {"eq_fraction": 0.9},
            {"ion_fraction": -0.1},
            {"ion_fraction": 2},
            {"ion_task_max": 3, "ion_fraction": 0.2},
        ):
            job["selection"] = invalid
            with self.assertRaises(ValueError):
                plasma.selection_options(job)

    def test_fractional_selection_caps_total_and_redistributes(self):
        task = self.make()
        self.run_task(task)
        job = self.data["model_devi_jobs"][0]
        job["selection"] = {
            "eq_fraction": 0.0,
            "ion_fraction": 0.4,
            "neutral_fraction": 0.5,
            "validation_fraction": 0.1,
            "max_frames_per_event": 20,
            "min_time_separation": 0.0,
        }
        totals = []
        for budget in (4, 10, 20):
            _, selected, _, _, metadata = plasma.select_frames(
                [str(task)], job, self.data, 0, budget
            )
            self.assertLessEqual(len(selected), budget)
            self.assertTrue(
                all(metadata[tuple(ref)]["kind"] != "eq" for ref in selected)
            )
            totals.append(len(selected))
        self.assertGreater(totals[-1], totals[0])
        # The ion-only reservation is unavailable, so eligible neutral events
        # borrow it rather than leaving otherwise usable FP slots unfilled.
        frames = json.loads((task / "frames.json").read_text())
        rows = np.atleast_2d(np.loadtxt(task / "model_devi.out"))
        for row in rows:
            if frames[str(int(row[0]))]["kind"] == "ion":
                row[4] = 0.4
        np.savetxt(task / "model_devi.out", rows)
        job["selection"] = {
            "eq_fraction": 0,
            "ion_fraction": 0.9,
            "neutral_fraction": 0.1,
            "validation_fraction": 0,
            "max_frames_per_event": 20,
            "min_time_separation": 0,
        }
        _, selected, _, _, metadata = plasma.select_frames(
            [str(task)], job, self.data, 0, 4
        )
        self.assertEqual(len(selected), 4)
        self.assertTrue(
            all(metadata[tuple(ref)]["kind"] == "neutral" for ref in selected)
        )

    def test_single_ion_event_remains_available_for_training(self):
        job = self.data["model_devi_jobs"][0]
        job["plasma"].update(nions=1, ratio=2)
        job["selection"] = {
            "eq_fraction": 0.1,
            "ion_fraction": 0.3,
            "neutral_fraction": 0.5,
            "validation_fraction": 0.1,
        }
        task = self.make()
        self.run_task(task)
        _, selected, _, _, metadata = plasma.select_frames(
            [str(task)], job, self.data, 0, 30
        )
        training = [ref for ref in selected if not metadata[tuple(ref)]["validation"]]
        self.assertTrue(any(metadata[tuple(ref)]["kind"] == "ion" for ref in training))
        self.assertFalse(
            any(
                metadata[tuple(ref)]["kind"] == "ion"
                and metadata[tuple(ref)]["validation"]
                for ref in selected
            )
        )

    def test_incar_selected_by_actual_snapshot_species(self):
        inputs = Path("incars")
        inputs.mkdir()
        (inputs / "Cu").write_text("ISMEAR = 1\nSIGMA = 0.2\n")
        (inputs / "Cu-O").write_text("ISMEAR = 0\nSIGMA = 0.05\n")
        (inputs / "ordinary").write_text("ISMEAR = -5\n")
        self.data["fp_incar"] = _abspath_fp_incar(
            {
                "19": {"Cu": "incars/Cu", "Cu-O": "incars/Cu-O"},
                "20": {"Cu": "incars/Cu", "default": "incars/Cu-O"},
                "default": "incars/ordinary",
            }
        )
        for system, oxygen, task_index in (
            (19, 0, 0),
            (19, 1, 1),
            (19, 4, 2),
            (20, 0, 0),
            (20, 2, 1),
            (10, 1, 0),
        ):
            task = Path(f"iter.000000/02.fp/task.{system:03d}.{task_index:06d}")
            task.mkdir(parents=True)
            (task / "frame.dump").write_text(snapshot(0, 0, oxygen))
            dpdata.System(
                str(task / "frame.dump"), fmt="lammps/dump", type_map=["Cu", "O"]
            ).to_vasp_poscar(str(task / "POSCAR"))
            expected = inputs / (
                "ordinary" if system == 10 else "Cu-O" if oxygen else "Cu"
            )
            self.assertEqual(
                _select_vasp_incar_path(self.data, task), str(expected.resolve())
            )
        # Exercise selection after the FP writer changes working directory.
        make_fp_vasp_incar(0, self.data)
        for task in Path("iter.000000/02.fp").glob("task.*.*"):
            system = int(task.name.split(".")[1])
            species = dpdata.System(str(task / "POSCAR"), fmt="vasp/poscar").data[
                "atom_names"
            ]
            expected = inputs / (
                "ordinary" if system == 10 else "Cu-O" if "O" in species else "Cu"
            )
            self.assertEqual((task / "INCAR").read_text(), expected.read_text())
        self.data["fp_incar"]["19"].pop("Cu-O")
        with self.assertRaisesRegex(RuntimeError, "species 'Cu-O'"):
            _select_vasp_incar_path(self.data, "iter.000000/02.fp/task.019.000001")

    def test_example_survives_schema_normalization(self):
        from dpgen.generator.arginfo import run_jdata_arginfo
        from dpgen.util import normalize

        repository = Path(__file__).resolve().parents[1]
        data = json.loads(
            (repository / "tests/generator/param-mg-vasp.json").read_text()
        )
        data["fp_incar"] = "INCAR"
        data["model_devi_jobs"] = [
            json.loads((repository / "examples/run/plasma/job.json").read_text())
        ]
        data["model_devi_restart_from_last"] = True
        normalized = normalize(run_jdata_arginfo(), data, strict_check=False)
        self.assertEqual(normalized["model_devi_jobs"][0]["plasma"]["nions"], 10)
        self.assertEqual(
            normalized["model_devi_jobs"][0]["selection"]["validation_fraction"], 0.1
        )

    def test_generated_sampling_uses_lammps_scalar_clamp(self):
        config = dict(
            plasma.DEFAULTS,
            temp=293,
            seed=1,
            trj_freq=10,
            stop_factor=2,
            f_hi=0.3,
            v_hi=1e10,
            models=["a", "b"],
        )
        for phase, frequency in (("eq", config["eq_trj_freq"]), ("exposure", 10)):
            state = worker.initial_state(config)
            state.update(phase=phase, step=975)
            text = worker.block_input(
                config, state, None, "checkpoints/test", None, 5.0, 0
            )
            self.assertNotIn("equal min(", text)
            self.assertNotIn("equal max(", text)
            self.assertIn("variable sample_steps equal v_checkpoint_end-step", text)
            self.assertIn(
                f'if "${{sample_steps}} > {frequency}" then "variable sample_steps equal {frequency}"',
                text,
            )
            self.assertIn("$(step) >= ${checkpoint_end}", text)

    def test_checkpoint_targets_do_not_restart_event_interval(self):
        config = dict(
            plasma.DEFAULTS,
            temp=293,
            seed=1,
            trj_freq=10,
            stop_factor=2,
            f_hi=0.3,
            v_hi=1e10,
            models=["a", "b"],
        )
        state = worker.initial_state(config)
        state.update(
            phase="exposure", eq_complete=True, time=17.2, active_event=28, arrivals=28
        )
        planned, insertion, target = worker.plan(config, state)
        self.assertIsNone(insertion)
        self.assertAlmostEqual(target, 5 + 28 * 5 / 11)
        self.assertEqual(planned["arrivals"], 28)
