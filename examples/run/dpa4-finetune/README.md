# DPA4 foundation fine-tuning

Start from an existing DP-GEN run parameter file for your target system, such as
`examples/run/ch4/param.json`. Set these top-level options:

```json
{
  "finetune_model_type": "dpa4",
  "finetune_model": "/absolute/path/to/pretrained.pt",
  "finetune_model_source": "previous",
  "numb_models": 4,
  "train_backend": "pytorch",
  "dp_compress": false
}
```

Replace `default_training_param` with the checkpoint's matching DeePMD training
configuration, including its complete `model` section (`type_map`, `descriptor`,
and `fitting_net`). Retain the foundation model's complete element ordering;
keep the top-level DP-GEN `type_map` and `mass_map` specific to your system.
DP-GEN fills in training data paths and assigns independent random seeds to the
ensemble members. Adjust the learning rate and training steps for fine-tuning.
A single foundation checkpoint is reused for all `numb_models` members; you can
instead supply a list containing one checkpoint per member.

Use `train_command: "dp"` or `"dp --pt"` in the machine configuration, with a
DeePMD-kit installation supporting DPA4. Run:

```bash
dpgen finetune param.json machine.json
```

The first iteration uses `--finetune`; subsequent iterations initialize from
the preceding training checkpoints. Set `finetune_model_source: "foundation"`
to start each iteration from the foundation instead. Interrupted tasks restart
from their local checkpoints.

DPA4 exports `frozen_model.pt2`, linked as `graph.000.pt2`, etc. Exploration uses
the system element symbols in `pair_coeff` and enables the LAMMPS atom map. Use
a compatible DeePMD/LAMMPS build and export on hardware suitable for inference.
Model compression is unsupported and is rejected. This example targets ordinary
energy/force full-model fine-tuning; spin and LoRA workflows are not validated.

See the [DeePMD DPA4 guide](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html)
for matching checkpoint configurations, backend requirements, and export details.
