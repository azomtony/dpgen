# Cu oxidation with plasma deposition

This mode adds energetic and thermal oxygen arrivals to the ordinary training →
exploration → single-point VASP labeling loop. It works with `dpgen run` and
`dpgen finetune`; existing NVT/NPT jobs are unchanged. It describes energetic
neutral O atoms with an atomic Cu–O potential, not explicit electronic plasma.

Start from your existing Cu/O run or fine-tuning parameter file. Set:

```json
{
  "type_map": ["Cu", "O"],
  "mass_map": [63.546, 15.999],
  "model_devi_engine": "lammps",
  "model_devi_restart_from_last": true,
  "model_devi_merge_traj": false,
  "numb_models": 4
}
```

Copy `job.json` from this directory into `model_devi_jobs`. Add one entry for
each intended exploration iteration. Keep your checkpoint-matching training
configuration, initial datasets, `sys_configs`, force/virial trust settings,
`fp_task_max`, and existing single-point VASP settings. With fractional selection,
`fp_task_max` is the per-system cap on **all** labels, including validation.
At least two models are required for committee disagreement.

Use the same Slurm machine configuration for training, exploration, and FP as
usual, requesting one GPU per job if desired. For example:

```bash
dpgen finetune param.json machine.json
```

Omit `--gpus` when DP-GEN should submit separate scheduler jobs. The exploration
command remains your configured `lmp` command. Its environment must provide
Python 3.9+ and a LAMMPS build supporting your DeePMD model backend, deposition,
evaporation, Langevin thermostatting, adaptive timesteps, and `fix halt`. No
LAMMPS Python bindings are needed. The controller is copied into each task.
Models are loaded together using the normal DeePMD committee pair style.

## Physical controls

The initial structure must be rectangular with periodic x/y, vacuum along z,
Cu type 1 and O type 2. The worker uses `boundary p p f`, fixes the bottom two Cu
planes using `surface` and `a`, and follows the supplied deposition protocol:

- Equilibrate mobile Cu for `eq_time` ps on a new trajectory branch.
- Thermostat moving atoms in the side strips of width `edge` during exposure.
  The strips extend up to the highest Cu atom plus 2 Å.
- Deposit one energetic O, then `ratio` thermal O atoms, repeated `nions` times.
- `eion` is the energetic O kinetic energy in eV; `temps` is the bath and neutral
  velocity-sampling temperature in K.
- `ion_dt` is the physical time in ps allocated to each energetic-plus-neutral
  group. Each individual arrival gets `ion_dt / (1 + ratio)` ps.
- Insert into the central footprint, inset by `edge + margin`, at `height` Å
  above the highest Cu atom. Keep the original near-overlap protection and stop
  with a LAMMPS error if there is insufficient insertion area or vacuum.
- Remove oxygen near the upper/lower z boundaries. `initial_dt`, `max_dt` are in
  ps, and `max_displacement` is in Å.

`nions` always requests **new** energetic arrivals, whatever the starting frame.
The default fresh branch has 5 ps equilibration and 50 ps exposure, with 10 new
energetic and 100 new neutral arrivals. Physical boundaries are checked every
integration step and may be exceeded by at most an integration step.

`surface` can be a scalar shared by all selected systems, or a list with exactly
one orientation per `sys_idx`, in the same order. For example:

```json
"sys_idx": [19, 20, 21],
"plasma": {
  "surface": [100, 110, 111]
}
```

This assigns Cu(100) to system 19, Cu(110) to system 20, and Cu(111) to system 21.
The surface list is a per-system mapping, **not** a sweep. A mismatched list length
or unsupported orientation raises an error. Each task stores its resolved scalar
orientation, which is also used when matching trajectories for continuation.

`temps` can contain multiple temperatures. Other scalar numeric `plasma` settings
can also be lists to generate a Cartesian sweep, for example `"eion": [5.0, 10.0]`.
A scalar optional `seed` supplies reproducible task seeds; different iterations
and tasks receive different derived seeds. Without it, task creation generates
fresh seeds and persists them for recovery.

## Sampling and stopping

`eq_trj_freq` and `trj_freq` are integration-step frequencies. Equilibration is
saved coarsely; all exposure intervals are saved frequently. Committee deviation
and individual `traj/STEP.lammpstrj` frames use the same frequency. Frames include
velocities, forces, physical time, and their actual variable atom count.

The worker checks deviation between short LAMMPS runs, at intervals no longer
than the current phase's sampling frequency. A failed frame above an upper trust
limit does not by itself stop the task. Reaching
`plasma_stop_factor * model_devi_f_trust_hi`, or the corresponding virial stop
limit, ends the task as `trust_stop`. Nonfinite deviations also stop it. The
ordinary accurate/candidate/failed thresholds do not change. Detection is
sampled, not a continuous-time guarantee. An invalid LAMMPS state is an error;
the previous complete checkpoint remains available.

Each saved frame has a `frames.json` entry with physical time, phase, arrival
number/type, time since arrival, equilibration progress, and cumulative incident
counts. `deposition.json` records arrival velocities, energies, and timestamps.

## Continuation and recovery

With `model_devi_restart_from_last: true`, a new iteration matches the immediately
preceding task by system, original structure, temperature, incident energy,
neutral ratio, arrival spacing, surface orientation/lattice spacing, insertion
geometry, and equilibration duration. It searches backward for a readable frame
below both upper trust limits and preserves that frame's **velocities**.

A deposition frame starts the new dose directly, without another equilibration.
A frame inside equilibration finishes only the remaining equilibration time.
If there is no usable matching frame, a new branch starts from the initial slab.
A changed exposure condition starts a separate branch. Changes to `nions` do not
prevent matching: the dose is new for each segment. Turning continuation off
starts a new branch for each task.

`checkpoint_freq` is the maximum number of integration steps between binary
checkpoints. Event boundaries can checkpoint earlier. Each immutable checkpoint
contains its event progress, absolute physical-time targets, restart checksum,
and sampled outputs. The restart and outputs are written before the state file
that commits them. The latest two binary checkpoints are retained.

Re-executing the **same worker command in the same task directory** resumes the
latest complete checksum-valid checkpoint. It never repeats a committed arrival,
never adds another equilibration, and never resets an event's remaining time.
Uncommitted work is replayed from the preceding checkpoint. Damaged latest
checkpoints fall back to the previous one. If all published checkpoints are
unusable, recovery fails explicitly rather than silently repeating the dose.
Completed and trust-stopped tasks are idempotent.

For Slurm preemption, scheduler requeue or dispatcher resubmission must rerun the
worker in the preserved work directory. The worker does not itself change Slurm
policy, requeue a job, or recover from a scheduler deleting its filesystem.

LAMMPS is restarted at checkpoint/event boundaries; the model committee remains
loaded during the intervening sampling checks. This trades model-loading
overhead for portable executable-based recovery. Arrival randomness is derived
from the saved seed and event. Langevin thermostat restarts are statistically
consistent but not bitwise identical: LAMMPS does not serialize that fix's RNG.
See the [LAMMPS Langevin restart documentation](https://docs.lammps.org/fix_langevin.html).
The requested dose and physical-time targets remain unchanged.

## Event-balanced labels and independent checks

Selection budgets are fractions of each system's `fp_task_max`:

```json
"selection": {
  "eq_fraction": 0.1,
  "ion_fraction": 0.3,
  "neutral_fraction": 0.5,
  "validation_fraction": 0.1,
  "max_frames_per_event": 3,
  "min_time_separation": 0.01
}
```

The four fractions must sum to 1. With `fp_task_max: 30`, the initial allocation
is 3 equilibration, 9 energetic-impact, 15 neutral-arrival training labels, and
3 validation labels. At 100, it becomes 10, 30, 50, and 10. Integer rounding uses
largest remainders so allocations sum to the budget, even for small budgets.
The total includes held-out labels and never exceeds `fp_task_max` per system.
It is independent of the overall accurate-frame fraction and `fp_task_min`.

If an enabled phase has too few eligible candidates, unused slots are borrowed
by other enabled phases, preferring exposure. Unused validation slots can also
be used for training. A phase with fraction zero remains disabled. Candidate
availability, `max_frames_per_event`, and `min_time_separation` still limit actual
counts; a larger budget does not force labeling redundant or failed frames.
Selection visits different arrivals before taking more frames from an arrival,
caps each event at `max_frames_per_event`, and enforces `min_time_separation` in
ps within an event. An event runs from insertion to the next insertion. Within
it, higher force-deviation candidates receive priority.

The validation fraction reserves an event-diverse set of accurate or candidate
frames, using priorities independent of disagreement. Exposure is preferred over
equilibration. Entire selected validation events are excluded from training
selection for this iteration. At least one candidate event per enabled phase is
reserved for training, so a segment's only energetic impact cannot be withheld
entirely. Very short exposures can therefore produce fewer held-out labels.

For compatibility, explicit `eq_task_max`, `ion_task_max`, `neutral_task_max`, or
`validation_task_max` settings select the old absolute-budget mode, where the
training cap and additional validation budget retain their previous behavior.
Do not mix absolute-budget keys and fraction keys in one selection block.

All chosen structures use the existing unrelaxed VASP single-point workflow.
Different Cu/O compositions are collected into separate `MultiSystems` groups.
POTCAR is assembled from each snapshot's actual species, including Cu-only
frames from equilibration. To choose different smearing settings for Cu-only
and oxygen-containing snapshots of the **same system**, use a nested `fp_incar`
entry (repeat for systems 20 and 21 as needed):

```json
"fp_incar": {
  "19": {
    "Cu": "input_fp/INCAR_cu",
    "Cu-O": "input_fp/INCAR_cuo"
  },
  "default": "input_fp/INCAR_cu"
}
```

Keep any existing system-index entries alongside these. Species keys are
alphabetically sorted element symbols joined by `-`; selection reads each FP
task's POSCAR. Any positive oxygen count selects `Cu-O`, regardless of overall
stoichiometry. This is an explicit input-file rule, not an inference of oxide
phase or electronic character. A nested `default` is optional; without one,
unmatched species raise an error. Ordinary string and system-index mappings,
and per-system KPOINTS settings, retain their existing behavior. Relative paths
are resolved from the workflow directory before entering FP task directories.

Training labels go into `02.fp/data.SSS`; validation labels go into
`02.fp/validation.SSS`, outside the training-data discovery paths. The FP failure
count includes both categories. Keep these validation directories excluded if
manually collecting or adding datasets later.

Use the existing model-testing command to compare the exploration committee and
the next retrained committee on the **same** held-out data, for example:

```bash
dpgen test-model iter.000000/02.fp iter.000000/00.train validation-before \
  --system-prefix validation. --model-pattern 'graph*.pt2' --dp-command 'dp --pt'
dpgen test-model iter.000000/02.fp iter.000001/00.train validation-after \
  --system-prefix validation. --model-pattern 'graph*.pt2' --dp-command 'dp --pt'
```

Use the matching model pattern/backend for your model, and `--machine` when
model tests should use batch resources. Validation is deliberately evaluated on
demand; it does not add an unrequested GPU evaluation stage to the main loop.

Trajectory cleanup retains at least the latest iteration for continuation and
cleans old plasma checkpoint archives along with old trajectory folders.
Currently this mode targets Cu/O, standard force/virial thresholds, separate
frames, and VASP single points; template inputs, PIMD, cluster selection,
electronic-temperature conditioning, and relative/adaptive trust options are
rejected rather than silently changing their meaning.

## Selection reports and FP origins

No extra option is required. Plasma FP preparation writes the following files
in `iter.XXXXXX/02.fp` for each system (for example, system 19):

- `plasma-selection.019.log`: readable requested budgets, observed phase rates,
  actual training/validation counts and FP shares, event counts, and FP origins.
- `plasma-selection.019.json`: the same report in structured form, including
  candidate shares, requested versus actual allocation, maximum force/virial
  deviations, and counts by source task and arrival event.
- `plasma-origins.019.csv`: one row per created FP task, with the source MD task,
  timestep, physical time, phase/event, time since arrival, temperature, surface,
  training/validation assignment, classification, and force/virial deviation.

The main DP-GEN log also prints each FP task's origin and a phase summary.
Each FP task's `job.json` retains its `plasma_frame` provenance, classification,
selection flag and deviation values. Training selections can differ from actual
created FP tasks when geometry filters reject a frame; the JSON keeps both
counts. Candidate/failure rates use **sampled frames** as the denominator, while
FP shares use **created FP tasks**. Sampling frequency differs between
equilibration and exposure, so raw frame counts are not physical-time rates.

The report distinguishes uncertainty from allocation: a stage with many failed
frames can receive few or no FP tasks because failed frames are not ordinary
training candidates. It reports exposure phases and arrival events. Locating
uncertainty within physical slab regions (core, edge bath, surface) requires
per-atom committee output and is not inferred from a global maximum deviation.

## Recover FP jobs without losing completed results

For Slurm VASP jobs, add this top-level section to `param.json`:

```json
"fp_recovery": {
  "enabled": true,
  "max_nonhardware_retries": 2,
  "hardware_retry_delay": 60,
  "nonhardware_retry_delay": 60,
  "check_interval": 30
}
```

All delays are seconds. Set `fp.resources.group_size` to `1` in `machine.json`.
Keep a persistent shared `remote_root` and use batch submission without `--gpus`.
The FP environment must provide Python 3.9+. This policy is opt-in; other stages
and ordinary submission behavior remain unchanged.

The controller first probes the original submission identity for an existing
recovery record. If one exists, it adopts the original jobs, job IDs, and staging
directory instead of wrapping or resubmitting them. Existing uninstrumented jobs
cannot retroactively provide GPU UUID telemetry; their scheduler and available
application diagnostics are still recorded. New FP stages use the instrumented
worker. Do not change commands, machine/resources settings or scientific inputs
while adopting an existing campaign. A previously abandoned submission under
*different* settings is not automatically discovered; restore its original
settings and recovery records before enabling recovery.

New campaigns skip validated local results. They persist a frozen task list and
submission identity in `02.fp/.fp-recovery.json`, reattach existing running jobs,
and download each finished task without waiting for siblings. Validation checks
normal VASP termination, electronic convergence, a single frame, matching atom
count/species/geometry, and finite energies and forces. Input fingerprints prevent
silently reusing results after changing POSCAR, INCAR, POTCAR or KPOINTS.

**Confirmed hardware failures retry without an attempt limit until successful
or the controller is cancelled.** Evidence includes Slurm `NODE_FAIL`, explicit
uncorrectable ECC errors and a GPU falling off the bus. Preemption, timeout,
OOM, convergence errors and unclassified CUDA/application errors do not become
unlimited hardware retries. Other failures get `max_nonhardware_retries` retries.
If those are exhausted, remaining jobs are still monitored and their valid
results preserved, then the stage raises a diagnostic error. This does not
silently enable partial-data training or weaken the existing failure threshold.

Each retry logs the job ID, available node/partition, failure category and reason.
Detailed evidence is appended to `02.fp/fp-recovery-history.jsonl`. Each newly
wrapped task also has `fp-attempt.json` and `fp-runtime.log`. Per-task status,
retry counters and the unchanged submission location remain in `.fp-recovery.json`.
Staging directories are retained. Multiple controllers for one FP stage are
prevented by an exclusive lock. An ambiguous `sbatch` outcome stops with a clear
message instead of risking duplicate submission; backend job-ID markers and
saved job IDs are checked before resubmitting.

For an instrumented GPU task, the worker records CUDA-visible device UUIDs and
preserves the scheduler allocation boundary. A confirmed single-device failure
is recorded in the workflow-root `.dpgen-gpu-health.json`. Later attempts mask
known-bad GPUs only when healthy GPUs are already in their allocation. A job
allocated only known-bad devices exits before VASP and requests another allocation.
The entire node is **not** automatically excluded. Remove an old manual
`#SBATCH --exclude=ghpc011` if you intend its healthy GPUs to remain eligible.

For numeric CUDA masks, UUID discovery uses the CUDA runtime when available;
explicit UUID masks are used directly. CUDA may remap indices, so Slurm/NVML
numeric IDs alone are never used to select another device. When mapping is
unavailable, the original scheduler mask is preserved and the limitation logged.
Multi-GPU error messages do not automatically blame every allocated GPU. Once a
physical device has been repaired, remove its entry from the GPU-health registry
before using it again. Allocation is still controlled by Slurm: the controller
can reject known-bad allocations but cannot force Slurm to assign a particular
healthy card, so eventual completion depends on healthy resources becoming
available.

### Explicitly choose an existing FP staging directory

If automatic recovery does not locate the existing submission, give the actual
staging directory on the configured FP machine:

```bash
dpgen finetune param.json machine.json \
  --fp-existing-dir /absolute/path/to/existing/submission-directory
```

The same option works with `dpgen run`. Supply the directory containing the
`<40-character-submission-hash>.json` recovery record and `task.*` directories,
not the parent containing several temporary directories. You can alternatively
supply the absolute path to that submission JSON file. The option implies FP
recovery for the selected stage even if `fp_recovery.enabled` was not set.

The record must match the current FP stage, task inputs, command and original
machine/resource settings. Inputs are checked before staging uploads. Recovery
restores the saved jobs and job IDs and pins every subsequent bind/upload to
this directory. A missing record, changed input, incompatible settings or a
record for another iteration produces an error; it never falls back to creating
a replacement directory. If another local recovery campaign may still have
active jobs, it refuses to silently abandon those jobs or create duplicates.

The chosen location persists in `02.fp/.fp-recovery.json`, so later invocations
with `fp_recovery.enabled: true` reuse it without repeating the flag. The CLI
selection applies only to the first FP stage reached by that invocation; later
iterations do not accidentally reuse the old staging directory. Resume while
`record.dpgen` still points to that iteration's FP work; this option does not
rewind completed workflow stages.
