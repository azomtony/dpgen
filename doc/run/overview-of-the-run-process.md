# Overview of the Run process

The run process contains a series of successive iterations, undertaken in order such as heating the system to certain temperatures. Each iteration is composed of three steps: exploration, labeling, and training. Accordingly, there are three sub-folders: 00.train, 01.model_devi, and 02.fp in each iteration.

00.train: DP-GEN will train several (default 4) models based on initial and generated data. The only difference between these models is the random seed for neural network initialization.

01.model_devi : represent for model-deviation. DP-GEN will use models obtained from 00.train to run Molecular Dynamics(default LAMMPS). Larger deviation for structure properties (default is the force of atoms) means less accuracy of the models. Using this criterion, a few structures will be selected and put into the next stage 02.fp for more accurate calculation based on First Principles.

02.fp : Selected structures will be calculated by first-principles methods(default VASP). DP-GEN will obtain some new data and put them together with initial data and data generated in previous iterations. After that, new training will be set up and DP-GEN will enter the next iteration!

In the run process of the DP-GEN, we need to specify the basic information about the system, the initial data, and details of the training, exploration, and labeling tasks. In addition, we need to specify the software, machine environment, and computing resource and enable the process of job generation, submission, query, and collection automatically. We can perform the run process as we expect by specifying the keywords in param.json and machine.json, and they will be introduced in detail in the following sections.

Here, we give a general description of the run process. We can execute the run process of DP-GEN easily by:

```sh
dpgen run param.json machine.json
```

The following files or folders will be created and upgraded by codes：

- iter.00000x contains the main results that DP-GEN generates in the first iteration.
- record.dpgen records the current stage of the run process.
- dpgen.log includes time and iteration information.

When the first iteration is completed, the folder structure of iter.000000 is like this:

```sh
$ ls iter.000000
00.train 01.model_devi 02.fp
```

In folder iter.000000/ 00.train:

- Folder 00x contains the input and output files of the DeePMD-kit, in which a model is trained.
- graph.00x.pb is the model DeePMD-kit generates. The only difference between these models is the random seed for neural network initialization.

In folder iter.000000/ 01.model_devi：

- Folder confs contains the initial configurations for LAMMPS MD converted from POSCAR you set in {dargs:argument}`sys_configs <run_jdata/sys_configs>` of param.json.
- Folder task.000.00000x contains the input and output files of the LAMMPS. In folder task.000.00000x, file model_devi.out records the model deviation of concerned labels, energy and force in MD. It serves as the criterion for selecting which structures and doing first-principle calculations.

In folder iter.000000/ 02.fp：

- candidate.shuffle.000.out records which structures will be selected from last step 01.model_devi. There are always far more candidates than the maximum you expect to calculate at one time. In this condition, DP-GEN will randomly choose up to {dargs:argument}`fp_task_max <run_jdata/fp_task_max>` structures and form the folder task.\*.
- rest_accurate.shuffle.000.out records the other structures where our model is accurate (`max_devi_f` is less than {dargs:argument}`model_devi_f_trust_lo <run_jdata[model_devi_engine=lammps]/model_devi_jobs/model_devi_f_trust_lo>`, no need to calculate any more),
- rest_failed.shuffled.000.out records the other structures where our model is too inaccurate (larger than {dargs:argument}`model_devi_f_trust_hi <run_jdata[model_devi_engine=lammps]/model_devi_jobs/model_devi_f_trust_hi>`, there may be some error).
- data.000: After first-principle calculations, DP-GEN will collect these data and change them into the format DeePMD-kit needs. In the next iteration's 00.train, these data will be trained together as well as the initial data.

DP-GEN identifies the stage of the run process by a record file, record.dpgen, which will be created and upgraded by codes. Each line contains two numbers: the first is the index of iteration, and the second, ranging from 0 to 9, records which stage in each iteration is currently running.

| Index of iterations | Stage in eachiteration | Process         |
| :------------------ | :--------------------- | :-------------- |
| 0                   | 0                      | make_train      |
| 0                   | 1                      | run_train       |
| 0                   | 2                      | post_train      |
| 0                   | 3                      | make_model_devi |
| 0                   | 4                      | run_model_devi  |
| 0                   | 5                      | post_model_devi |
| 0                   | 6                      | make_fp         |
| 0                   | 7                      | run_fp          |
| 0                   | 8                      | post_fp         |

0,1,2 correspond to make_train, run_train, post_train. DP-GEN will write scripts in make_train, run the task by specific machine in run_train and collect result in post_train. The records for model_devi and fp stage follow similar rules.

If the process of DP-GEN stops for some reasons, DP-GEN will automatically recover the main process by record.dpgen. You may also change it manually for your purpose, such as removing the last iterations and recovering from one checkpoint. When re-running dpgen, the process will start from the stage that the last line record.

## Training DPA4C models from scratch

`dpgen run` recognizes DPA4C from
`default_training_param.model.descriptor.type = "dpa4c"`. Supply your complete
DeePMD DPA4C model/training configuration under `default_training_param` and your
system's element list in `type_map`. No foundation checkpoint or
`finetune_model_type` setting is required. The PyTorch backend is selected when
`train_backend` is omitted; if supplied, it must be `"pytorch"`.

Use DeePMD-kit with the experimental PyTorch DPA4C backend and a matching LAMMPS
installation. Set `deepmd_version` to your installed version (for example `"3.2"`)
and `train.command` to `"dp --pt-expt"` in `machine.json`. If the command is simply
`"dp"`, DP-GEN adds `--pt-expt`. Conflicting backend flags are rejected.

```sh
dpgen run param.json machine.json --gpus 0 1 2 3
```

The regular workflow generates independently seeded models using your system's
type map, trains with `--skip-neighbor-stat`, and restarts interrupted training
from `model.ckpt.pt`. Existing `training_init_model`/`training_reuse_iter` options
can initialize subsequent training from previous checkpoints. Training from
scratch does not use `--finetune` or `--use-pretrain-script`.

Freezing uses `freeze -c model.ckpt.pt -o frozen_model --lower-kind graph`.
When `dp_compress` is enabled, compression produces `compressed_model.pt2`.
Exploration uses the resulting `graph.NNN.pt2` links, atom mapping, and explicit
element names in the LAMMPS `pair_coeff` command. Interactive command environment
overrides and completion tracking also apply. Omit `--gpus` to use the existing
batch dispatcher with the same DPA4C command generation.

For foundation-model fine-tuning, continue to use `dpgen finetune` and its DPA4C
configuration. Initialization from frozen/foundation models through the regular
DPA4C training path is rejected; use that dedicated fine-tuning workflow instead.

## Automatic validation holdout

For both `run` and `finetune`, add these top-level options to `param.json`:

```json
"validation_fraction": 0.1,
"validation_seed": 42
```

No machine-file change is needed. Zero (the default) disables splitting. The
fraction must be less than one. The workflow generates `training_data` and
`validation_data` in every model's input. Do not also provide explicit
`default_training_param.training.validation_data` when using this option.
DeePMD-kit 2.x or newer input format is required.

Each dataset reserves a seeded, contiguous block of approximately the requested
fraction of frames (rounded up, while retaining at least one training frame).
Single-frame datasets remain training-only with a warning; if no validation
frames can be reserved, preparation fails. All committee members share the same
split. A contiguous block reduces interleaving of neighboring frames, but it is
not a trajectory-aware or independent-structure split. Use curated separate
validation data when stronger separation is needed.

The original datasets remain unchanged. Persistent split data and frame indices
are stored in `.dpgen-validation` in the workflow root. Preserve that directory
across restarts and iterations. Reused sources retain their assignments; newly
labeled datasets get their own split. Changed source contents, fraction, or seed
are rejected for an existing cached source: use a fresh workflow directory for
such changes. Per-iteration copies in `00.train/data.validation` are forwarded
by the batch dispatcher, including when `one_h5` combines the training data.

Validation uses batch size one and up to ten sampled validation batches per
display interval, controlled by the existing `training.disp_freq`. Thus it uses
a fixed held-out pool, not a full exhaustive validation pass at every report.
Training batch sizes are capped to the remaining frame counts when specified as
integers or lists. Other training, loss, and model settings are unchanged.

Enable this in a fresh workflow, with models that have not trained on the reserved
frames. Adding a split after training does not undo earlier exposure. Editing
`param.json` during an interrupted training stage does not regenerate existing
`input.json` files. No changes to batch size or learning rate are required merely
to enable validation.

## Fine-tuning DPA4 foundation models

Use `dpgen finetune param.json machine.json` with
`finetune_model_type: "dpa4"`, `finetune_model` pointing to a `.pt` or `.pth`
checkpoint, and `dp_compress: false`. One checkpoint can initialize all
`numb_models` ensemble members. DPA4 uses the standard PyTorch backend (`dp --pt`),
not DPA4C's experimental backend; conflicting machine `train_command` backend
flags are rejected.

Supply the checkpoint's matching complete model configuration under
`default_training_param.model`, including its full `type_map`, DPA4/SeZM
`descriptor`, and `fitting_net`. DP-GEN preserves the model's element ordering
while using the top-level system `type_map` for exploration. All system elements
must occur in the model element map. Training data paths and ensemble seeds are
assigned by DP-GEN. Tune learning rate and training steps for your target data.

The workflow fine-tunes from the foundation in iteration zero and initializes
from the previous iteration's checkpoints thereafter, unless
`finetune_model_source: "foundation"` is selected. Export produces `.pt2` archives
for exploration, which requires a compatible DeePMD/LAMMPS installation. DPA4
compression is unsupported. The supported path is ordinary energy/force
full-model fine-tuning; LoRA and spin workflows have not been validated.

See `examples/run/dpa4-finetune/README.md` for a configuration recipe.

## Training without D3, exploring with D3

When the DFT labels include Grimme D3 (for example VASP `IVDW=12`, D3 with
Becke–Johnson damping), the model can learn only the DFT part. Exploration then
adds D3 back through LAMMPS. The two options are independent, and both are off
by default:

```json
"training_strip_d3": {
  "damping_function": "bj",
  "functional": "pbe",
  "cutoff": 50.2,
  "cn_cutoff": 20.0
},
"lmp_d3": {
  "enable": true,
  "damping_function": "bj",
  "functional": "pbe",
  "cutoff": 50.2,
  "cn_cutoff": 20.0
}
```

`training_strip_d3` works with `dpgen run` and `dpgen finetune`. At every
`make_train`:

1. Each training system (initial data and every iteration's labels) is run once
   through LAMMPS `pair_style dispersion/d3`, frame by frame.
2. The D3 energy, forces and virial are subtracted.
3. The result is cached in `./.dpgen-d3`. The cache is rebuilt only when the
   labels or the D3 settings change.

Training reads the copies from `00.train/data.nod3`. The labels on disk keep
the D3 term. Explicit `validation_data` systems are stripped too, and
`validation_fraction` splits the stripped data. Every system needs element
names, from `type_map.raw`.

D3 is computed by the `d3_command` in machine.json (default `lmp`), run where
`dpgen` runs. That LAMMPS must include `dispersion/d3` (check with
`lmp -h | grep dispersion`). It can be an MPI launcher, for example
`"d3_command": "mpirun -np 8 lmp"`.

If both blocks are enabled, their settings must match. Otherwise the model
would see D3 counted twice or missing, so dpgen stops with an error. Match the
DFT settings too:

- VASP `IVDW=12` is `"bj"`, and `IVDW=11` is `"original"` (zero damping).
- `cutoff` and `cn_cutoff` correspond to `VDW_RADIUS` and `VDW_CNRADIUS`
  (VASP defaults: 50.2 and 20 Å).
- LAMMPS `dispersion/d3` has no three-body term, which matches VASP's default.

Model deviation in `model_devi.out` comes only from the deepmd models, so trust
levels are not affected by D3. Labels in `02.fp/validation.*` still include D3;
`dpgen analyze parity` strips them before comparing.

## Held-out FP test set and parity plots

To keep part of each iteration's single points out of training as test data,
set:

```json
"fp_holdout_fraction": 0.1,
"fp_holdout_seed": 42
```

How tasks are held out:

- Each new FP task is held out with that probability (at most 0.5). The draw is
  seeded per iteration and per system.
- Each system keeps at least one new label for training.
- Held-out tasks carry an `fp-holdout` marker file. `post_fp` writes their
  labels to `02.fp/validation.SSS` instead of `data.SSS`, so no model ever
  trains on them.
- This is VASP only. Plasma jobs keep their own holdout.

`dpgen analyze parity` merges the held-out labels from all iterations into one
fixed test set. It then tests the models of the selected iterations against it:

```bash
dpgen analyze parity -i 5                  # one iteration
dpgen analyze parity -i 2-6                # one row of plots per iteration
dpgen analyze parity -i 2-6 --model mean   # committee-mean prediction
dpgen analyze parity -i 2-6 --machine machine.json   # run dp test as a job
```

The output goes to `analysis/`:

- `parity.png`: one row per iteration. Each row has energy per atom and force
  components, model against DFT. All systems are in each panel, colored by
  `sys_configs` entry, with RMSE, MAE and frame count in each panel.
- `parity_summary.csv`: RMSE and MAE per iteration, overall and per system.

`--model` selects a model index (default 0) or `mean`. `dp test` results are
reused until the test set or the model changes, so adding iterations to the
range only tests the new ones. With `training_strip_d3`, the test set is
stripped of D3 the same way, so models are compared with matching labels. The
D3 command comes from `--d3-command`, or the machine file's `d3_command`.


### Continue exploration from a previous trajectory

For native classical LAMMPS exploration in either `dpgen run` or
`dpgen finetune`, set this top-level option in `param.json`:

```json
"model_devi_restart_from_last": true
```

Each new task looks for a task in the immediately preceding iteration with the
same system index, original starting-structure path, ensemble, temperature,
and pressure. Electronic temperature and PKA energy must also match when used.
Matching uses conditions rather than task numbering, so reordering the list of
temperatures does not change which trajectory is continued. Existing native
tasks without restart metadata can be matched through their original structure
symlinks.

DP-GEN searches saved frames backward in timestep order, choosing the latest
readable frame whose maximum force and virial deviations are both strictly below
their upper trust limits. Accurate and candidate frames qualify; equality with
an upper limit, nonfinite deviations, and failed frames do not. Selection uses
the preceding iteration's trust settings (including per-system and per-job
limits), `model_devi_skip`, and force normalization when enabled. The usual
virial upper-limit default is `1e10`. Both separate `traj/*.lammpstrj` files and
merged `all.lammpstrj` trajectories are supported.

If there is no matching task or no acceptable saved frame, the task starts from
its original structure. DP-GEN writes the source task and timestep, or the
fallback reason, into `job.json` under `restart`, and logs the decision.
Coordinates and the simulation cell are continued; velocities are initialized
afresh by the normal exploration input.

With this option enabled, trajectory cleanup retains at least the latest
iteration so it can supply the next iteration's starting frames. A longer
`model_devi_clean_traj` retention setting or `false` is still respected.
Trajectories removed before enabling the option cannot be recovered. The option
applies when exploration tasks are created; it does not rewind already completed
tasks. It defaults to `false`, preserving the original starting-structure and
cleanup behavior. Template inputs, PIMD, and cluster selection are not supported
with this option.

### Pulling two molecules together (aggregation bias)

To sample how two molecules aggregate, for example two asphaltenes whose
aromatic cores stack, add `aggregation_bias` to a `model_devi_jobs` entry.
LAMMPS needs the COLVARS package.

```json
{
  "sys_idx": [0],
  "ensemble": "nvt",
  "temps": [300],
  "nsteps": 200000,
  "trj_freq": 100,
  "aggregation_bias": {
    "cores": "auto",
    "target": 3.5,
    "k": 1.0,
    "pull_steps": 100000,
    "recenter": true
  }
}
```

Each exploration task then:

- **Pulls the cores together.** The Colvars `distance` between the centers of
  mass of the two cores is pulled by a moving harmonic restraint, from its
  starting value to `target` (Å) over `pull_steps`. After that it is held at
  `target`.
  - `k` is in eV/Å².
  - `pull_steps` defaults to half of `nsteps`.
  - `upper_wall` (Å) adds a wall that stops the cores drifting back apart.
- **Keeps the pair centered** (`recenter`, on by default). `fix recenter`
  translates the whole system every step so the pair's center sits at the box
  center. It adds no force and keeps the pair away from the periodic
  boundaries.
- **Leaves orientation free.** The cores can approach face-to-face, tilted or
  edge-on.

Selecting the cores:

- `"cores": "auto"` finds molecules from covalent bonds, then the largest fused
  system of sp2 ring atoms in each one. Molecules whose core has at least
  `min_core_atoms` atoms (default 10) count. This excludes benzene and toluene
  solvent rings, which have 6. Exactly two must be found.
- Explicit cores are two lists of 0-based atom indices in POSCAR order:
  `[[...], [...]]`. Use a dict keyed by system index to give each system its
  own pair. Explicit indices cannot be combined with `shuffle_poscar`.

Each task writes its own `conf.lmp` with image flags, so both cores are whole
and the second core is the periodic image nearest the first. The cores must
start less than half the box width apart, otherwise the pull could follow a
periodic image; dpgen stops with an error in that case. The pulling state is
kept in LAMMPS restart files, so a resumed task continues the pull.

`job.json` records the cores (with element counts) and the starting distance.
`colvars.colvars.traj`, written every `trj_freq` steps, lists the core distance
for each dumped frame. Frames are still selected by model deviation; the bias
only steers where the trajectory goes. The bias works only with native
classical LAMMPS jobs (no templates, plasma, PIMD or `model_devi_nopbc`).

### Plasma deposition exploration

Use `"ensemble": "plasma"` in a LAMMPS `model_devi_jobs` entry for Cu/O energetic
and thermal oxygen deposition. Controls include separate equilibration/exposure
sampling, committee stopping, velocity-preserving continuation, transactional
checkpoints, event-balanced FP budgets, and held-out validation events. See the
[plasma configuration recipe](../../examples/run/plasma/README.md) and its
`job.json` for settings and recovery behavior. Labeling continues to use the
existing VASP single-point workflow.
