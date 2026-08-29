#!/usr/bin/env python3

"""Fine-tuning workflow for DP-GEN.

This module keeps the standard ``dpgen run`` workflow untouched while exposing a
small wrapper that starts from PyTorch DeePMD fine-tuning and then reuses the
existing exploration and FP stages.
"""

import argparse
import copy
import glob
import itertools
import json
import logging
import logging.handlers
import os
import queue
import shlex

from dpgen import SHORT_CMD, dlog


DPA4C_TYPE_MAP = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

_FINETUNE_MODEL_TYPES = {"dp3", "dpa4c"}
_DEEPMD_BACKEND_FLAGS = {"--pt", "--pt-expt", "--jax"}
DPA4C_FROZEN_MODEL = "frozen_model.pt2"
DPA4C_COMPRESSED_MODEL = "compressed_model.pt2"


def prepare_finetune_jdata(jdata):
    """Return a fine-tune-ready copy of the run parameters."""
    jdata = copy.deepcopy(jdata)

    finetune_model_type = jdata.get("finetune_model_type", "dp3")
    if finetune_model_type not in _FINETUNE_MODEL_TYPES:
        raise RuntimeError(
            "finetune_model_type should be either 'dp3' or 'dpa4c'."
        )
    jdata["finetune_model_type"] = finetune_model_type

    if "finetune_model" in jdata:
        if (
            "training_finetune_model" in jdata
            and jdata["training_finetune_model"] != jdata["finetune_model"]
        ):
            raise RuntimeError(
                "finetune_model and training_finetune_model are both set but differ."
            )
        jdata["training_finetune_model"] = jdata["finetune_model"]

    models = jdata.get("training_finetune_model")
    if isinstance(models, str):
        models = [models]
    if not models:
        raise RuntimeError(
            "dpgen finetune requires training_finetune_model, or finetune_model, "
            "to point to PyTorch .pt or .pth models."
        )
    models = [os.path.expanduser(model) for model in models]
    jdata["training_finetune_model"] = models

    if jdata.get("training_init_model", False):
        raise RuntimeError("training_init_model cannot be used with dpgen finetune.")
    if jdata.get("training_init_frozen_model") is not None:
        raise RuntimeError(
            "training_init_frozen_model cannot be used with dpgen finetune."
        )

    jdata.setdefault("train_backend", "pytorch")
    if jdata["train_backend"] != "pytorch":
        raise RuntimeError("dpgen finetune currently supports train_backend='pytorch'.")

    if jdata.get("mlp_engine", "dp") != "dp":
        raise RuntimeError("dpgen finetune currently supports mlp_engine='dp'.")
    model_suffixes = {os.path.splitext(model)[1] for model in models}
    valid_model_suffixes = {".pt", ".pth"}
    bad_suffix = [
        model
        for model in models
        if os.path.splitext(model)[1] not in valid_model_suffixes
    ]
    if bad_suffix:
        raise RuntimeError(
            "dpgen finetune expects .pt or .pth model files: "
            + ", ".join(bad_suffix)
        )
    if len(model_suffixes) != 1:
        raise RuntimeError("all fine-tune model files should use the same suffix.")
    jdata["finetune_model_suffix"] = model_suffixes.pop()

    model_source = jdata.get("finetune_model_source", "previous")
    if model_source not in ("previous", "foundation"):
        raise RuntimeError(
            "finetune_model_source should be either 'previous' or 'foundation'."
        )
    jdata["finetune_model_source"] = model_source

    model_branch = jdata.get("finetune_model_branch")
    if model_branch is not None:
        if not isinstance(model_branch, str) or not model_branch:
            raise RuntimeError("finetune_model_branch should be a non-empty string.")
        jdata["finetune_model_branch"] = model_branch

    numb_models = jdata.get("numb_models")
    if (
        finetune_model_type == "dpa4c"
        and numb_models is not None
        and len(models) == 1
        and numb_models > 1
    ):
        models = models * numb_models
        jdata["training_finetune_model"] = models
    if numb_models is not None and len(models) != numb_models:
        raise RuntimeError(
            "training_finetune_model should contain exactly numb_models entries."
        )

    missing_models = [model for model in models if not os.path.isfile(model)]
    if missing_models:
        raise FileNotFoundError(
            "Cannot find fine-tune model file(s): " + ", ".join(missing_models)
        )

    if jdata.get("training_reuse_iter") is None or jdata["training_reuse_iter"] < 1:
        jdata["training_reuse_iter"] = 1

    if finetune_model_type == "dpa4c":
        jdata["dp_train_skip_neighbor_stat"] = True

    if _uses_pretrain_script(jdata):
        finetune_args = jdata.get("finetune_args", "")
        if "--use-pretrain-script" not in finetune_args.split():
            finetune_args = f"{finetune_args} --use-pretrain-script".strip()
        jdata["finetune_args"] = finetune_args

    return jdata


def _uses_pretrain_script(jdata):
    if jdata.get("finetune_model_type") == "dpa4c":
        return True
    model = jdata.get("default_training_param", {}).get("model", {})
    return model.get("descriptor") == {} or model.get("fitting_net") == {}


def _get_finetune_training_type_map(jdata):
    if jdata.get("finetune_model_type") == "dpa4c":
        return DPA4C_TYPE_MAP
    return jdata["type_map"]


def _get_train_backend_flag(jdata):
    if jdata.get("finetune_model_type") == "dpa4c":
        return "--pt-expt"
    return "--pt"


def _get_train_command(jdata, mdata):
    train_command = mdata.get("train_command", "dp").strip()
    train_backend_flag = _get_train_backend_flag(jdata)
    command_tokens = shlex.split(train_command)
    existing_backend_flags = _DEEPMD_BACKEND_FLAGS.intersection(command_tokens)
    if existing_backend_flags:
        if (
            train_backend_flag == "--pt-expt"
            and "--pt-expt" not in existing_backend_flags
        ):
            raise RuntimeError(
                "finetune_model_type='dpa4c' requires train_command to use "
                "'--pt-expt'."
            )
        return train_command
    return f"{train_command} {train_backend_flag}"


def _get_frozen_model_suffix(jdata):
    if jdata.get("finetune_model_type") == "dpa4c":
        return ".pt2"
    return ".pth"


def _get_freeze_command(jdata, train_command):
    if jdata.get("finetune_model_type") == "dpa4c":
        return (
            f"{train_command} freeze -c model.ckpt.pt -o frozen_model "
            "--lower-kind graph"
        )
    return f"{train_command} freeze"


def _get_compress_command(jdata, train_command):
    if jdata.get("finetune_model_type") == "dpa4c":
        return (
            f"{train_command} compress -i {DPA4C_FROZEN_MODEL} "
            f"-o {DPA4C_COMPRESSED_MODEL}"
        )
    return f"{train_command} compress"


def _get_finetune_args(jdata, include_model_branch):
    args = jdata.get("finetune_args", "")
    if include_model_branch and jdata.get("finetune_model_branch"):
        args = f"{args} --model-branch {jdata['finetune_model_branch']}".strip()
    return args


def _get_init_model_name(jdata):
    return f"init{jdata.get('finetune_model_suffix', '.pth')}"


def _make_train_finetune(iter_index, jdata, mdata, link_foundation):
    from dpgen.generator.lib.utils import make_iter_name
    from dpgen.generator.run import (
        default_train_input_file,
        make_train,
        train_name,
        train_task_fmt,
    )

    make_jdata = copy.deepcopy(jdata)
    make_jdata.pop("training_finetune_model", None)
    model = make_jdata["default_training_param"].get("model", {})
    if model.get("descriptor") == {}:
        model.pop("descriptor")

    make_train(iter_index, make_jdata, mdata)
    if link_foundation:
        _link_finetune_models(iter_index, jdata)

    if not _uses_pretrain_script(jdata):
        return

    input_model = copy.deepcopy(jdata["default_training_param"]["model"])
    input_model["type_map"] = _get_finetune_training_type_map(jdata)
    work_path = os.path.join(make_iter_name(iter_index), train_name)
    for ii in range(jdata["numb_models"]):
        input_path = os.path.join(
            work_path, train_task_fmt % ii, default_train_input_file
        )
        if not os.path.isfile(input_path):
            continue
        with open(input_path) as fp:
            train_input = json.load(fp)
        train_input["model"] = copy.deepcopy(input_model)
        with open(input_path, "w") as fp:
            json.dump(train_input, fp, indent=4)


def _link_finetune_models(iter_index, jdata):
    from dpgen.generator.run import train_name, train_task_fmt
    from dpgen.generator.lib.utils import create_path, make_iter_name

    work_path = os.path.join(make_iter_name(iter_index), train_name)
    for ii, model in enumerate(jdata["training_finetune_model"]):
        task_old_path = os.path.join(work_path, train_task_fmt % ii, "old")
        create_path(task_old_path)
        target = os.path.abspath(model)
        link_name = os.path.join(task_old_path, _get_init_model_name(jdata))
        for stale_name in ("init.pt", "init.pth"):
            stale_path = os.path.join(task_old_path, stale_name)
            if os.path.lexists(stale_path):
                os.remove(stale_path)
        os.symlink(os.path.relpath(target, task_old_path), link_name)


def _run_train_pytorch_with_init(iter_index, jdata, mdata, init_from_foundation):
    from packaging.version import Version

    from dpgen.dispatcher.Dispatcher import make_submission
    from dpgen.generator.lib.utils import check_api_version
    from dpgen.generator.run import (
        default_train_input_file,
        fp_name,
        make_iter_name,
        set_version,
        train_name,
        train_task_fmt,
    )
    from dpgen.util import expand_sys_str

    if jdata.get("mlp_engine", "dp") != "dp":
        raise ValueError(f"Unsupported engine: {jdata.get('mlp_engine')}")

    numb_models = jdata["numb_models"]
    train_input_file = default_train_input_file
    suffix = _get_frozen_model_suffix(jdata)

    if "srtab_file_path" in jdata.keys():
        zbl_file = os.path.basename(jdata.get("srtab_file_path", None))

    try:
        mdata["deepmd_version"]
    except KeyError:
        mdata = set_version(mdata)

    train_command = _get_train_command(jdata, mdata)
    finetune_args = _get_finetune_args(jdata, init_from_foundation)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, train_name)
    copy_flag = os.path.join(work_path, "copied")
    if os.path.isfile(copy_flag):
        dlog.info("copied model, do not train")
        return

    all_task = []
    for ii in range(numb_models):
        task_path = os.path.join(work_path, train_task_fmt % ii)
        all_task.append(task_path)

    commands = []
    if Version("1") <= Version(mdata["deepmd_version"]) < Version("4"):
        extra_flags = ""
        if jdata.get("dp_train_skip_neighbor_stat", False):
            extra_flags += " --skip-neighbor-stat"
        command = f"{train_command} train {train_input_file}{extra_flags}"
        if init_from_foundation:
            init_name = _get_init_model_name(jdata)
            init_flag = f"--finetune old/{init_name} {finetune_args}".strip()
        else:
            init_flag = f"--init-model old/model.ckpt {finetune_args}".strip()
        command = (
            "{ if [ ! -f model.ckpt.pt ]; then "
            f"{command} {init_flag}; "
            f"else {command} --restart model.ckpt; fi }}"
        )
        commands.append(f"/bin/bash -c {shlex.quote(command)}")
        commands.append(_get_freeze_command(jdata, train_command))
        if jdata.get("dp_compress", False):
            commands.append(_get_compress_command(jdata, train_command))
    else:
        raise RuntimeError(
            "DP-GEN currently only supports for DeePMD-kit 1.x to 3.x version!"
        )

    run_tasks = [os.path.basename(ii) for ii in all_task]

    forward_files = [train_input_file]
    if "srtab_file_path" in jdata.keys():
        forward_files.append(zbl_file)
    if init_from_foundation:
        forward_files.append(os.path.join("old", _get_init_model_name(jdata)))
    else:
        forward_files.append(os.path.join("old", "model.ckpt.pt"))

    backward_files = [
        f"frozen_model{suffix}",
        "lcurve.out",
        "train.log",
        "checkpoint",
        "model.ckpt.pt",
    ]
    if jdata.get("dp_compress", False):
        if jdata.get("finetune_model_type") == "dpa4c":
            backward_files.append(DPA4C_COMPRESSED_MODEL)
        else:
            backward_files.append(f"frozen_model_compressed{suffix}")

    if not jdata.get("one_h5", False):
        init_data_sys = [
            os.path.join("data.init", ii) for ii in jdata["init_data_sys"]
        ]
        trans_comm_data = []
        cwd = os.getcwd()
        os.chdir(work_path)
        fp_data = glob.glob(os.path.join("data.iters", "iter.*", fp_name, "data.*"))
        for ii in itertools.chain(init_data_sys, fp_data):
            sys_paths = expand_sys_str(ii)
            for single_sys in sys_paths:
                if "#" not in single_sys:
                    trans_comm_data += glob.glob(os.path.join(single_sys, "set.*"))
                    trans_comm_data += glob.glob(os.path.join(single_sys, "type*.raw"))
                    trans_comm_data += glob.glob(os.path.join(single_sys, "nopbc"))
                else:
                    trans_comm_data.append(single_sys.split("#")[0])
    else:
        cwd = os.getcwd()
        trans_comm_data = ["data.hdf5"]
    trans_comm_data = list(set(trans_comm_data))
    os.chdir(cwd)

    try:
        train_group_size = mdata["train_group_size"]
    except Exception:
        train_group_size = 1

    user_forward_files = mdata.get("train" + "_user_forward_files", [])
    forward_files += [os.path.basename(file) for file in user_forward_files]
    backward_files += mdata.get("train" + "_user_backward_files", [])

    check_api_version(mdata)

    submission = make_submission(
        mdata["train_machine"],
        mdata["train_resources"],
        commands=commands,
        work_path=work_path,
        run_tasks=run_tasks,
        group_size=train_group_size,
        forward_common_files=trans_comm_data,
        forward_files=forward_files,
        backward_files=backward_files,
        outlog="train.log",
        errlog="train.log",
    )
    submission.run_submission()


def run_finetune_iter(param_file, machine_file):
    """Run fine-tuning followed by the existing exploration and FP stages."""
    from dpgen.generator.arginfo import run_jdata_arginfo
    from dpgen.generator.lib.utils import log_iter, make_iter_name, record_iter
    from dpgen.generator.run import (
        make_fp,
        make_model_devi,
        post_fp,
        post_model_devi,
        post_train,
        run_fp,
        run_model_devi,
        run_train,
        update_mass_map,
    )
    from dpgen.remote.decide_machine import convert_mdata
    from dpgen.util import load_file, normalize, sepline, setup_ele_temp

    jdata = prepare_finetune_jdata(load_file(param_file))

    mdata = load_file(machine_file)

    jdata_arginfo = run_jdata_arginfo()
    wrapper_jdata = {
        key: jdata[key]
        for key in (
            "finetune_args",
            "finetune_model_type",
            "finetune_model_branch",
            "finetune_model_source",
            "finetune_model_suffix",
            "training_finetune_model",
        )
        if key in jdata
    }
    normalized_input = copy.deepcopy(jdata)
    normalized_input.pop("finetune_model_source", None)
    normalized_input.pop("finetune_model", None)
    jdata = normalize(jdata_arginfo, normalized_input, strict_check=False)
    jdata.update(wrapper_jdata)
    jdata = prepare_finetune_jdata(jdata)

    update_mass_map(jdata)

    use_ele_temp = jdata.get("use_ele_temp", 0)
    if use_ele_temp == 1:
        setup_ele_temp(False)
    elif use_ele_temp == 2:
        setup_ele_temp(True)

    if jdata.get("pretty_print", False):
        from monty.serialization import dumpfn

        fparam = (
            SHORT_CMD
            + "_"
            + param_file.split(".")[0]
            + "."
            + jdata.get("pretty_format", "json")
        )
        dumpfn(jdata, fparam, indent=4)
        fmachine = (
            SHORT_CMD
            + "_"
            + machine_file.split(".")[0]
            + "."
            + jdata.get("pretty_format", "json")
        )
        dumpfn(mdata, fmachine, indent=4)

    if mdata.get("handlers", None):
        if mdata["handlers"].get("smtp", None):
            que = queue.Queue(-1)
            queue_handler = logging.handlers.QueueHandler(que)
            smtp_handler = logging.handlers.SMTPHandler(**mdata["handlers"]["smtp"])
            listener = logging.handlers.QueueListener(que, smtp_handler)
            dlog.addHandler(queue_handler)
            listener.start()

    mdata = convert_mdata(mdata)
    finetune_model_source = jdata["finetune_model_source"]
    max_tasks = 10000
    numb_task = 9
    record = "record.dpgen"
    iter_rec = [0, -1]
    if os.path.isfile(record):
        with open(record) as frec:
            for line in frec:
                iter_rec = [int(x) for x in line.split()]
        if len(iter_rec) == 0:
            raise ValueError("There should not be blank lines in record.dpgen.")
        dlog.info("continue from iter %03d task %02d" % (iter_rec[0], iter_rec[1]))  # noqa: UP031

    cont = True
    ii = -1
    while cont:
        ii += 1
        if ii < iter_rec[0]:
            continue
        iter_name = make_iter_name(ii)
        sepline(iter_name, "=")
        for jj in range(numb_task):
            if ii * max_tasks + jj <= iter_rec[0] * max_tasks + iter_rec[1]:
                continue
            task_name = "task %02d" % jj  # noqa: UP031
            sepline(f"{iter_name} {task_name}", "-")
            if jj == 0:
                log_iter("make_train", ii, jj)
                _make_train_finetune(
                    ii,
                    jdata,
                    mdata,
                    finetune_model_source == "foundation" or ii == 0,
                )
            elif jj == 1:
                log_iter("run_train", ii, jj)
                if finetune_model_source == "foundation" or ii == 0:
                    _run_train_pytorch_with_init(ii, jdata, mdata, True)
                elif _uses_pretrain_script(jdata):
                    _run_train_pytorch_with_init(ii, jdata, mdata, False)
                else:
                    run_train(ii, jdata, mdata)
            elif jj == 2:
                log_iter("post_train", ii, jj)
                post_train(ii, jdata, mdata)
            elif jj == 3:
                log_iter("make_model_devi", ii, jj)
                cont = make_model_devi(ii, jdata, mdata)
                if not cont:
                    break
            elif jj == 4:
                log_iter("run_model_devi", ii, jj)
                run_model_devi(ii, jdata, mdata)
            elif jj == 5:
                log_iter("post_model_devi", ii, jj)
                post_model_devi(ii, jdata, mdata)
            elif jj == 6:
                log_iter("make_fp", ii, jj)
                make_fp(ii, jdata, mdata)
            elif jj == 7:
                log_iter("run_fp", ii, jj)
                run_fp(ii, jdata, mdata)
            elif jj == 8:
                log_iter("post_fp", ii, jj)
                post_fp(ii, jdata)
            else:
                raise RuntimeError("unknown task %d, something wrong" % jj)  # noqa: UP031
            record_iter(record, ii, jj)


def gen_finetune(args):
    """CLI entry point for ``dpgen finetune``."""
    if args.PARAM and args.MACHINE:
        if args.debug:
            dlog.setLevel(logging.DEBUG)
        dlog.info("start fine-tuning")
        run_finetune_iter(args.PARAM, args.MACHINE)
        dlog.info("finished")


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("PARAM", type=str, help="The parameters of the generator")
    parser.add_argument(
        "MACHINE", type=str, help="The settings of the machine running the generator"
    )
    parser.add_argument("-d", "--debug", action="store_true", help="log debug info")
    args = parser.parse_args()
    gen_finetune(args)
