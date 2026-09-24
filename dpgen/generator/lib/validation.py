"""Persistent, disjoint validation splits for generated training inputs."""

import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import dpdata
import numpy as np

from dpgen import dlog


def validation_options(jdata):
    fraction = jdata.get("validation_fraction", 0.0)
    seed = jdata.get("validation_seed", 42)
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0 <= fraction < 1
    ):
        raise ValueError("validation_fraction must be in [0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("validation_seed must be a non-negative integer")
    if fraction and jdata.get("default_training_param", {}).get("training", {}).get(
        "validation_data"
    ):
        raise ValueError(
            "Use validation_fraction or explicit validation_data, not both"
        )
    return fraction, seed


def _split(source, cache, fraction, seed):
    system = dpdata.LabeledSystem(
        source, fmt="deepmd/hdf5" if "#" in source else "deepmd/npy"
    )
    nframes = len(system)
    if nframes == 0:
        raise ValueError(f"Cannot split empty training system: {source}")
    digest = hashlib.sha256()
    for name, value in sorted(system.data.items()):
        digest.update(name.encode())
        if isinstance(value, np.ndarray):
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.tobytes())
        else:
            digest.update(repr(value).encode())
    metadata = dict(
        source=source,
        fraction=fraction,
        seed=seed,
        frames=nframes,
        digest=digest.hexdigest(),
    )
    key = hashlib.sha256(source.encode()).hexdigest()[:24]
    destination = cache / key
    manifest = destination / "split.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if any(previous.get(k) != v for k, v in metadata.items()):
            raise ValueError(
                f"Validation source or split settings changed: {source}. Use a fresh workflow directory."
            )
        return destination, previous
    # Hold out a contiguous block rather than randomly interleaving nearby frames.
    count = min(nframes - 1, max(1, math.ceil(nframes * fraction)))
    rng = np.random.default_rng([seed, int(key, 16)])
    start = int(rng.integers(nframes - count + 1))
    validation = list(range(start, start + count))
    training = list(range(start)) + list(range(start + count, nframes))
    metadata.update(training=training, validation=validation)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        output = Path(temporary) / "split"
        output.mkdir()
        system.sub_system(training).to_deepmd_npy(str(output / "training"))
        if validation:
            system.sub_system(validation).to_deepmd_npy(str(output / "validation"))
        (output / "split.json").write_text(json.dumps(metadata, indent=2))
        output.rename(destination)
    return destination, metadata


def make_validation_split(input_files, work_path, jdata):
    """Write shared split datasets and configure all committee members identically."""
    fraction, seed = validation_options(jdata)
    if not fraction or not input_files:
        return
    work = Path(work_path).resolve()
    cache = Path.cwd() / ".dpgen-validation"
    first = Path(input_files[0]).resolve()
    template = json.loads(first.read_text())
    if "training_data" not in template["training"]:
        raise ValueError(
            "validation_fraction requires DeePMD-kit 2.x or newer training_data"
        )
    training_data = template["training"]["training_data"]
    train_paths, val_paths, train_counts = [], [], []
    total_train = total_val = 0
    for source in training_data["systems"]:
        file, separator, group = source.partition("#")
        source = str((first.parent / file).resolve()) + (
            separator + group if separator else ""
        )
        cached, manifest = _split(source, cache, fraction, seed)
        destination = work / "data.validation" / cached.name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(cached, destination)
        prefix = f"../data.validation/{cached.name}"
        train_paths.append(prefix + "/training")
        train_counts.append(len(manifest["training"]))
        total_train += len(manifest["training"])
        total_val += len(manifest["validation"])
        if manifest["validation"]:
            val_paths.append(prefix + "/validation")
        else:
            dlog.warning(
                "Validation: single-frame system remains training-only: %s", source
            )
    if not val_paths:
        raise ValueError("No validation frames: at least one system needs two frames")
    for filename in input_files:
        path = Path(filename)
        data = json.loads(path.read_text())
        training = data["training"]["training_data"]
        training["systems"] = train_paths
        sizes = training.get("batch_size")
        if isinstance(sizes, list):
            training["batch_size"] = [
                min(size, count) for size, count in zip(sizes, train_counts)
            ]
        elif isinstance(sizes, int):
            training["batch_size"] = [min(sizes, count) for count in train_counts]
        data["training"]["validation_data"] = {
            "systems": val_paths,
            "batch_size": 1,
            "numb_btch": min(10, total_val),
        }
        path.write_text(json.dumps(data, indent=4))
    dlog.info(
        "Validation split: %d training frames, %d validation frames; shared by %d models",
        total_train,
        total_val,
        len(input_files),
    )
