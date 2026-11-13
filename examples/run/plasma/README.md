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

The record must match the current FP stage, task inputs, command and machine
identity. Resource differences from `machine.json` are logged by field name;
the saved submission's resources remain authoritative for its existing jobs and
retries. Current resource settings apply to new submissions, not jobs being
resumed. Inputs are checked before staging uploads. Recovery
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


FP recovery prints a progress heartbeat every `check_interval`, including
successful, running, waiting, retry-wait, pending and failed counts. Before a
potentially slow operation it logs whether it is querying Slurm, collecting
outputs, validating results, staging pending inputs, or diagnosing a failure.
The local `.fp-recovery.json` also records the last activity and timestamp.
Already-submitted jobs do not trigger bulk input restaging during recovery;
inputs are staged for unsubmitted tasks or restored when a retry needs them.


## Hardware recovery for training and exploration

To enable the same hardware retry policy for training, fine-tuning, ordinary
LAMMPS exploration, plasma exploration, and Slurm VASP labeling, add this at the
**top level of `param.json`**, alongside `model_devi_jobs` and `fp_task_max`:

```json
"job_recovery": {
  "enabled": true,
  "max_nonhardware_retries": 2,
  "hardware_retry_delay": 60,
  "nonhardware_retry_delay": 60,
  "check_interval": 30
}
```

All delays are seconds. Existing `fp_recovery` settings override these defaults
for FP only. FP still requires `fp.resources.group_size: 1`. Training and
exploration automatically use one task per submitted job with this option,
including when their ordinary `group_size` is larger. Resource requests per job
(such as one GPU) still come from `machine.json`. Use Slurm batch execution,
without `--gpus`, and a persistent shared staging filesystem accessible from
replacement nodes. The compute environment needs Python 3.9+.

Confirmed hardware failures retry without a limit. Application, resource,
configuration, timeout, and unknown failures retain the bounded retry policy.
Reports include scheduler state, exit status, node, partition and available GPU
UUID evidence. The workflow shares `.dpgen-gpu-health.json` across stages;
masking only uses GPUs actually assigned by Slurm. If every allocated GPU is
known faulty, the attempt exits for another allocation rather than using an
unallocated GPU.

Rerun the same DP-GEN command with unchanged inputs and resource settings to
reattach existing recovery jobs. Successful siblings remain preserved. Training
uses its existing DeePMD checkpoint restart command. Ordinary LAMMPS uses its
existing restart files and `run ... upto` target; before the first checkpoint,
it must replay from the start. Plasma retains its event/dose checkpoint logic
and physical-time target. Retry handling does not upload initial checkpoint or
trajectory directories over newer remote state.

Existing ordinary training or exploration submissions are adopted when the
original submission record is found and each existing job contains one task.
Recovery keeps the original command, resources, job IDs, and staging directory;
it verifies that the staged inputs match before continuing. Those older jobs
have no wrapper receipt or GPU UUID telemetry, so success requires the saved
scheduler completion plus expected outputs, and failure diagnosis uses scheduler
records and the stage application log. New stages use the instrumented wrapper.
Existing jobs grouping multiple tasks are rejected rather than split into
potentially duplicate jobs. Keep the original inputs and machine settings.
The explicit `--fp-existing-dir` adoption option remains specific to FP. Other
exploration engines are not supported by `job_recovery` yet.

The controller checks a successful wrapper receipt matching the input digest
and nonempty exported model or deviation output. Plasma additionally requires
`complete` or `trust_stop` status. Normal scientific postprocessing still runs;
a wrapper receipt does not replace scientific validation. For compatibility,
shared bookkeeping files retain their `fp` names inside each stage directory:
`.fp-recovery.json`, `fp-recovery-history.jsonl`, and per-task `fp-attempt.json`
and `fp-runtime.log`. Progress messages identify the stage.

### CUDA context creation error 999

`cuCtxCreate ... error 999 (CUDA_ERROR_UNKNOWN)` is classified separately as
`gpu_initialization`. It retries without an attempt limit, using
`hardware_retry_delay`, across all recovery-enabled stages. It is not treated as
proof of a physically faulty GPU: driver or compute-mode problems can also
prevent context creation (see [NVIDIA's context creation documentation](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__CTX.html)).
Generic `CUDA_ERROR_UNKNOWN` messages elsewhere, CUDA out-of-memory, and illegal
memory accesses do not receive this special policy.

On restarting the controller with updated code, saved `unknown` diagnoses whose
captured evidence matches this pattern are reclassified. Exhausted matching tasks
become eligible for retry again, retaining the existing submission identity,
completed results and checkpoints. No manual ledger edits or staging-directory
deletion are needed. Reclassification is logged in the recovery history.

GPU initialization failures do not populate the confirmed-fault GPU blacklist.
In particular, when runtime UUID lookup is unavailable, do not infer a physical
GPU from differing `CUDA_VISIBLE_DEVICES` and `SLURM_JOB_GPUS` numeric values.
Retrying this failure does not guarantee avoidance of that physical GPU; device
avoidance still requires an authoritative UUID and confirmed hardware evidence.

### Identify the physical GPU used by an attempt

The worker records `node`, `allocated_gpus` (physical UUID, NVML index and PCI bus
ID), `selected_gpu_uuids`, `gpu_identity_source`, and the full `gpu_inventory` in
`fp-attempt.json`. Treat the inventory as a list of devices, not proof that all
of them were allocated to the job.

For numeric CUDA masks, the worker asks the CUDA runtime for each visible
ordinal's PCI bus ID with `cudaDeviceGetPCIBusId`, then uniquely matches that PCI
address to the NVML inventory. It normalizes PCI domain formatting and does not
assume that CUDA, Slurm and NVML numeric indices match. This replaces the earlier
lookup that incorrectly expected a `cudaDeviceGetUuid` runtime symbol. PCI-only
mapping is refused for enabled or unknown MIG mode, since a physical parent UUID
must not replace a MIG allocation's mask. Explicit scheduler-provided MIG UUIDs
remain supported.

Instrumentation is refreshed before submitting an instrumented retry, so an
existing recovery staging directory can receive the corrected worker without
changing scientific inputs, scheduler identity, or checkpoints. Already running
attempts are left alone. After updating the cluster code and restarting the
controller, inspect the next attempt's report, for example from `02.fp`:

```bash
python3 -m json.tool task.019.000007/fp-attempt.json
```

If CUDA enumeration is itself broken or the runtime library cannot be loaded,
identity remains unresolved and the scheduler's device mask is preserved. In
that case a cluster administrator can correlate the saved job ID and node with
the site's GRES-to-device mapping. On Slurm versions supporting it,
[`Flags=env_uuid` with AutoDetect](https://slurm.schedmd.com/gres.conf.html)
provides UUID-based `CUDA_VISIBLE_DEVICES`, allowing identity capture even when
CUDA cannot initialize. This is a cluster setting, not a `param.json` setting.
Identification of the assigned physical GPU is separate from proving that the
card is faulty; error 999 alone still does not trigger device blacklisting.

## Stop unproductive VASP single points

Add a top-level `fp_monitor` section to `param.json`. This example enables a
150-iteration budget and an optional stagnation heuristic:

```json
"fp_monitor": {
  "enabled": true,
  "max_scf_steps": 150,
  "min_scf_steps": 60,
  "stagnation_window": 20,
  "min_relative_improvement": 0.05,
  "unconverged_factor": 100,
  "startup_timeout": 1800,
  "stall_timeout": 1800,
  "check_interval": 10,
  "stop_grace_period": 120,
  "kill_grace_period": 10
}
```

This is opt-in and supports Slurm VASP single-point tasks (`NSW <= 1`, positive
`EDIFF`), using one FP task per job. Enabling it also activates the FP recovery
controller if it was not enabled already. The monitor runs inside the allocated
batch job and consumes the live VASP stdout that is recorded in `fp.log`, plus
new OSZICAR records. It does not assume that nonconvergence proves bad geometry:
SCF algorithm, mixing and smearing settings can also be responsible.

The criteria are:

- Nonfinite electronic energy changes or residuals request a stop immediately.
- `max_scf_steps` limits electronic iterations without EDIFF convergence; the
  INCAR `NELM` limit is also respected. Setting this option to zero removes the
  additional budget but retains the NELM check. The default extra cap is 200.
- `stagnation_window: 20` compares the best energy error in two successive
  20-step windows, after at least `min_scf_steps`. The error is the larger of
  absolute `dE` and `d eps`. A stop is requested only when the newer window fails
  to improve by `min_relative_improvement` and its best error remains above
  `unconverged_factor * EDIFF`. Improving or near-converged windows continue.
  This is a configurable heuristic, not proof that a calculation could never
  converge. The default `stagnation_window` is zero (heuristic disabled).
- `startup_timeout` limits waiting for the first recognized electronic iteration.
- `stall_timeout` limits inactivity in stdout, OSZICAR and OUTCAR after electronic
  iterations begin. Growing OUTCAR/OSZICAR output counts as activity, even when
  a slow electronic step has not finished. Zero disables either timeout.
- Optional `max_runtime` sets an overall attempt time budget; its default is zero
  (disabled). All time options are seconds.

Stopping first writes [`LABORT = .TRUE.` to STOPCAR](https://vasp.at/wiki/STOPCAR),
which requests termination at the next electronic step. If VASP does not exit
within `stop_grace_period`, the worker signals only its own application process
group with SIGTERM, then SIGKILL after `kill_grace_period`. It does not cancel
other tasks or jobs. The usual Slurm cleanup still applies when a batch job ends.

The reason and last SCF record are preserved in `fp-monitor-stop.json`,
`fp-attempt.json`, the application log and recovery history. Recovery reports
these tasks as `discarded` / `scientific_stop`, never resubmits them unchanged,
and continues monitoring their siblings. Their partial outputs are excluded
from both training and validation labeling, even when a parser could read an
energy or force. The structures and diagnostic outputs remain available for
review or separate recalculation with revised settings. Such stops count toward
the existing `failed_ratio` / `ratio_failed` limit; the workflow can still stop
at postprocessing if too few valid labels remain. This monitor does not relax
that quality check or change INCAR settings.

New submissions and subsequent instrumented retries receive the monitor.
Already running scripts are unchanged, and adopted older uninstrumented
submissions cannot acquire the compute-node watchdog retroactively. Updating a
controller alone does not start monitoring an already running VASP process.
Scientific commands, input fingerprints, job IDs and staging identities remain
unchanged when this policy is enabled for an existing instrumented campaign.

## Discard selected FP jobs that are already running

Use `dpgen fp-discard` when inspection shows that particular FP inputs should
not be run or retried unchanged. This works for recovery-managed running jobs,
including older jobs that do not have the SCF watchdog. It is a separate manual
decision from automatic SCF monitoring.

First update the cluster installation and restart the DP-GEN recovery controller
with the same workflow command. Stopping only that controller leaves its Slurm
jobs intact. An older running controller does not know how to consume these new
requests. From the affected iteration's `02.fp` directory, for example:

```bash
dpgen fp-discard . --jobs 43349984 --reason "SCF oscillating without convergence"
```

Or select the input task, including any current retry, rather than an old job ID:

```bash
dpgen fp-discard . --tasks task.021.000037 --reason "SCF oscillating without convergence"
```

Multiple job IDs or task names can be supplied. Add `--dry-run` to preview the
resolved task list without writing requests or cancelling anything. Unknown IDs,
tasks outside this stage, and already validated successful tasks are rejected.
A supplied job ID resolves to a task in the local recovery manifest. If that
same input is requeued before the controller reads the request, cancellation
follows the task's current job ID; it does not target an unrelated allocation.
Existing grouped jobs are rejected rather than cancelling unselected tasks.

The CLI queues an atomic, locked `.fp-discard-requests.json` request in `02.fp`;
it does not directly contact Slurm. The updated controller consumes it at its
next check. If the controller is stopped, resume it to carry out cancellation.
Before calling `scancel` through the configured machine, the controller persists
`discard_pending` in its ledger and writes `fp-discard.json` in each selected
task directory. This disables retries even if cancellation fails or the
controller exits between operations. It then waits for scheduler confirmation
before marking the task `discarded`. Pending requests and cancellation errors
remain visible in the queue/ledger and controller logs.

The request is scoped to the submission hash and scientific-input fingerprint.
Changed identities cause an error without scheduler mutation. Restarting the
controller resumes pending cancellations and does not automatically resubmit
the selected tasks. The reason is recorded in recovery history. The scoped
queue intent and discard marker exclude outputs from both training and holdout
labels, including a task finishing between request and cancellation. Structures,
logs and any partial results are retained. Other tasks continue normally.
Discarded tasks still count toward the existing FP failure-ratio quality check.

### Changing wall time while resuming training or exploration

Changing `machine.json` resource fields, including `custom_flags` wall-time
requests, does not require abandoning an existing recovery campaign. Recovery
verifies the task input fingerprints, scientific commands, transfer lists,
machine/context, and saved submission identity before restoring the original
submission record. It then uses the **saved resources** for its existing jobs
and retries. The controller reports the changed resource field names; the
current `machine.json` resources apply when a new stage is created. This does
not update the Slurm time limit of an already submitted job.

This also supports older stage recovery manifests that stored only a signature:
input fingerprints and the saved submission/worker policy are checked before
resuming. The original signature, staging directory, job IDs, and checkpoints
remain intact. Changed scientific inputs, commands, common-data/model file
lists, or machine/context settings are still rejected with a specific message.
If the original submission record cannot be found or verified, recovery fails
rather than creating a replacement directory. Do not delete the recovery
manifest or checkpoints to bypass the guard.
