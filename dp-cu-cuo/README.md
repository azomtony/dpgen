# Cu/CuO DP-GEN Project

This directory is the project layer for building a DeePMD potential for Cu and
Cu/O systems with DP-GEN, LAMMPS, and VASP.

Keep the upstream DP-GEN source unchanged unless we find a workflow gap that
cannot be solved with configuration or helper scripts.

## Intended HPC Layout

Create the run directory on a scratch/work filesystem:

```bash
mkdir -p /scratch/$USER/dp-cu-cuo
cd /scratch/$USER/dp-cu-cuo
```

Recommended contents on the HPC:

```text
dp-cu-cuo/
├── initial_data/          # existing labeled DeepMD data
├── seeds/                 # seed POSCAR files for exploration configs
├── generated/sys_configs/ # generated POSCARs for model deviation starts
├── pseudopotentials/      # POTCAR_Cu and POTCAR_O, or site-specific links
├── param.json
├── machine.json
├── run_dpgen.sh
├── record.dpgen
├── dpgen.log
└── iter.000000/
```

## First Setup Pass

1. Copy this `dp-cu-cuo` directory to the HPC.
2. Put existing labeled DeepMD data under `initial_data/`.
3. Put seed structures under `seeds/`, for example:

```text
seeds/
├── cu_fcc/POSCAR
├── cuo_bulk/POSCAR
└── cu_cuo_interface/POSCAR
```

4. Generate exploration starting structures:

```bash
python scripts/generate_exploration_configs.py \
  --seed-dir seeds \
  --output generated/sys_configs \
  --supercell 2 2 2 \
  --scales 0.96 0.98 1.00 1.02 1.04 \
  --n-pert 20 \
  --rattle 0.03 \
  --strain 0.03
```

5. Edit `configs/project.json` for your HPC paths and training settings.
6. Render DP-GEN config:

```bash
python scripts/render_param.py configs/project.json param.json
```

7. Fill in `machine.json` with real HPC queues, executable paths, and activation
   commands.
8. Validate:

```bash
python scripts/validate_dpgen_inputs.py param.json machine.json
```

9. Run from an activated DP-GEN environment:

```bash
bash run_dpgen.sh
```

## Initial Training Only

Because we already have initial labeled data, the first useful HPC action is to
train the initial model ensemble.

Prepare the train-only machine file:

```bash
# machine.train.json is already provided as a local editable copy.
# To reset it, copy:
cp configs/machine.template.train-only.json machine.train.json
```

Edit `machine.train.json` for the real DeePMD activation command, queue, GPU
count, and command.

Then run:

```bash
bash run_initial_train.sh
```

This creates and submits:

```text
iter.000000/00.train/000
iter.000000/00.train/001
iter.000000/00.train/002
iter.000000/00.train/003
```

After successful training, `post_train` links the frozen models under
`iter.000000/00.train/graph.000.pb`, etc. The script records progress through
`0 2` in `record.dpgen`, so the later full active-learning run can continue at
`make_model_devi`.

## Train And Explore, But Do Not Submit FP

The editable `machine.json` intentionally has no `fp` machine block. This lets
DP-GEN train, explore, select candidates, and prepare `02.fp/task.*`, but it
cannot accidentally submit VASP jobs.

Run:

```bash
bash run_until_make_fp.sh
```

This stops after `iter.XXXXXX/02.fp/task.*` has been created. Then inspect or
overwrite:

```text
iter.XXXXXX/02.fp/task.*/INCAR
iter.XXXXXX/02.fp/task.*/KPOINTS
```

before submitting single-point calculations.

You can check the task-specific files with:

```bash
python scripts/check_fp_profiles.py param.json iter.000000/02.fp
```

## Per-System VASP INCAR

This DP-GEN checkout supports per-system VASP INCAR selection. In `param.json`,
`fp_incar` may be a string, list, or dict. Our project uses a dict:

```json
"fp_incar": {
  "0": "templates/INCAR.cu-metal",
  "1": "templates/INCAR.oxide",
  "default": "templates/INCAR.static"
}
```

The keys `0`, `1`, etc. refer to the `sys_configs` index. DP-GEN writes the
chosen INCAR into each `iter.XXXXXX/02.fp/task.<sys_idx>.*` directory. KPOINTS
are then generated from that task's INCAR, so each profile can use a different
`KSPACING` / `KGAMMA` rule.

## Per-System VASP KPOINTS

This DP-GEN checkout also supports explicit per-system VASP KPOINTS through
`fp_kpoints`. Use this when you need exact Monkhorst-Pack meshes instead of a
spacing rule:

```json
"fp_kpoints": {
  "0": {
    "style": "Monkhorst-Pack",
    "mesh": [2, 2, 1],
    "shift": [0, 0, 0]
  },
  "1": {
    "style": "Monkhorst-Pack",
    "mesh": [2, 1, 2],
    "shift": [0, 0, 0]
  }
}
```

If a system index is missing from `fp_kpoints`, DP-GEN falls back to the normal
`KSPACING` / `fp_aniso_kspacing` behavior for that task.

## Notes For Cu/CuO

- Use `type_map` order `["Cu", "O"]` everywhere.
- Keep POTCAR order consistent with `type_map`: Cu first, O second.
- Include enough seed structures to cover the final use case. Bulk-only data will
  not make a good surface, interface, defect, or redox potential.
- If VASP single-point jobs must run on another cluster, use
  `configs/machine.template.split-fp-cluster.json` as the starting point.
