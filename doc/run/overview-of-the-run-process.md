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
