"""Remove the D3 dispersion term from labels before training.

DFT labels computed with D3 (for example VASP ``IVDW=12``) contain
``E_DFT + E_D3``. With ``training_strip_d3`` the models learn only ``E_DFT``,
and ``lmp_d3`` adds D3 back during exploration. Both must use the same D3
settings, so they are checked against each other.

The D3 term is computed with LAMMPS ``pair_style dispersion/d3`` on every
frame, the same code exploration uses. Labels on disk are never changed: each
source system is written once, stripped, to ``./.dpgen-d3/<key>`` and copied
into ``00.train/data.nod3``, and the training inputs point there.
"""

import hashlib
import json
import math
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import dpdata
import numpy as np
from dpdata.periodic_table import ELEMENTS

from dpgen import dlog

CACHE = ".dpgen-d3"
WORK_DIR = "data.nod3"
DAMPING = ("original", "zerom", "bj", "bjm")
PARAMS = ("damping_function", "functional", "cutoff", "cn_cutoff")
# LAMMPS metal units: pressure (bar) * volume (Å^3) / NKTV2P = energy (eV).
NKTV2P = 1.6021765e6
# Bump when the computation changes, so cached copies are rebuilt.
VERSION = 1


def strip_options(jdata):
    """Return the D3 settings to strip, or None when stripping is off."""
    block = jdata.get("training_strip_d3")
    if not block or not block.get("enable", True):
        return None
    missing = [key for key in PARAMS if key not in block]
    if missing:
        raise ValueError(f"training_strip_d3 is missing {missing}")
    params = {key: block[key] for key in PARAMS}
    if params["damping_function"] not in DAMPING:
        raise ValueError(f"training_strip_d3.damping_function must be one of {DAMPING}")
    if not isinstance(params["functional"], str) or not params["functional"]:
        raise ValueError("training_strip_d3.functional must be a functional name")
    for key in ("cutoff", "cn_cutoff"):
        value = params[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"training_strip_d3.{key} must be a positive number")
        params[key] = float(value)
    explore = jdata.get("lmp_d3") or {}
    if explore.get("enable", False):
        different = [
            key
            for key in PARAMS
            if str(explore.get(key)) != str(params[key])
            and not (
                key in ("cutoff", "cn_cutoff")
                and isinstance(explore.get(key), (int, float))
                and float(explore[key]) == params[key]
            )
        ]
        if different:
            raise ValueError(
                "lmp_d3 and training_strip_d3 must use the same D3 settings; "
                f"they differ in {different}"
            )
    return params


def lammps_input(atom_names, nframes, params, periodic):
    """LAMMPS script computing D3 energy, forces and virial for frame.N.data."""
    boundary = "p p p" if periodic else "f f f"
    names = " ".join(atom_names)
    style = (
        f"{params['damping_function']} {params['functional']} "
        f"{params['cutoff']:.17g} {params['cn_cutoff']:.17g}"
    )
    return f"""variable i loop {nframes}
label frame
clear
units metal
atom_style atomic
atom_modify map yes
boundary {boundary}
read_data frame.${{i}}.data
mass * 1.0
pair_style dispersion/d3 {style}
pair_coeff * * {names}
compute vir all pressure NULL virial
thermo_style custom step pe vol c_vir[1] c_vir[2] c_vir[3] c_vir[4] c_vir[5] c_vir[6]
thermo_modify format float %.17g
dump forces all custom 1 forces.${{i}} id fx fy fz
dump_modify forces sort id format float %.17g
run 0
print "${{i}} $(pe:%.17g) $(vol:%.17g) $(c_vir[1]:%.17g) $(c_vir[2]:%.17g) $(c_vir[3]:%.17g) $(c_vir[4]:%.17g) $(c_vir[5]:%.17g) $(c_vir[6]:%.17g)" append d3.out screen no
undump forces
next i
jump SELF frame
"""


def _write_frame(path, cell, coords, types, ntypes):
    """Write a LAMMPS data file for a lower-triangular cell."""
    lines = [
        "D3 frame written by dpgen",
        "",
        f"{len(types)} atoms",
        f"{ntypes} atom types",
        "",
        f"0 {cell[0][0]:.17g} xlo xhi",
        f"0 {cell[1][1]:.17g} ylo yhi",
        f"0 {cell[2][2]:.17g} zlo zhi",
        f"{cell[1][0]:.17g} {cell[2][0]:.17g} {cell[2][1]:.17g} xy xz yz",
        "",
        "Atoms # atomic",
        "",
    ]
    lines += [
        f"{index} {kind + 1} {x:.17g} {y:.17g} {z:.17g}"
        for index, (kind, (x, y, z)) in enumerate(zip(types, coords), 1)
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def _read_forces(path, natoms):
    lines = Path(path).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
    columns = lines[start].split()[2:]
    rows = np.array(
        [line.split() for line in lines[start + 1 :] if line.strip()], dtype=float
    )
    if rows.shape != (natoms, len(columns)):
        raise RuntimeError(f"Unexpected D3 force dump {path}")
    rows = rows[np.argsort(rows[:, columns.index("id")])]
    return rows[:, [columns.index(name) for name in ("fx", "fy", "fz")]]


def _run_lammps(command, workdir):
    result = subprocess.run(
        [*shlex.split(command), "-in", "in.d3", "-log", "log.d3", "-screen", "none"],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        log = Path(workdir, "log.d3")
        tail = log.read_text().splitlines()[-15:] if log.exists() else []
        raise RuntimeError(
            f"D3 LAMMPS run failed ({command}): "
            + "\n".join(tail or [result.stderr.strip()])
        )


def compute_d3(system, params, command, runner=_run_lammps):
    """Return D3 energies, forces and virials for every frame of ``system``.

    Results are in the system's own frame: LAMMPS needs a lower-triangular
    cell, so each frame is rotated for LAMMPS and the forces and virial are
    rotated back.
    """
    names = list(system["atom_names"])
    unknown = [name for name in names if name not in ELEMENTS]
    if unknown:
        raise ValueError(
            f"D3 needs element names, got {unknown}; add type_map.raw to the data"
        )
    nframes, natoms = system.get_nframes(), system.get_natoms()
    periodic = not system.nopbc
    rotated = system.copy()
    turns = []
    with tempfile.TemporaryDirectory(prefix="dpgen-d3-") as workdir:
        for index in range(nframes):
            coords = rotated["coords"][index]
            if periodic:
                turns.append(rotated.rot_frame_lower_triangular(index))
                cell = rotated["cells"][index]
            else:
                # An open box just large enough; D3 ignores the cell here.
                turns.append(np.eye(3))
                low = coords.min(axis=0)
                coords = coords - low + params["cutoff"]
                cell = np.diag(coords.max(axis=0) + params["cutoff"])
            _write_frame(
                Path(workdir, f"frame.{index + 1}.data"),
                cell,
                coords,
                system["atom_types"],
                len(names),
            )
        Path(workdir, "in.d3").write_text(
            lammps_input(names, nframes, params, periodic)
        )
        runner(command, workdir)
        table = np.atleast_2d(np.loadtxt(Path(workdir, "d3.out")))
        if table.shape != (nframes, 9) or not np.isfinite(table).all():
            raise RuntimeError("D3 LAMMPS run did not report every frame")
        table = table[np.argsort(table[:, 0])]
        energies = table[:, 1]
        forces = np.empty((nframes, natoms, 3))
        virials = np.empty((nframes, 3, 3))
        for index in range(nframes):
            turn = turns[index]
            # x' = x @ turn, so vectors go back with turn.T and tensors with
            # turn @ W' @ turn.T.
            forces[index] = (
                _read_forces(Path(workdir, f"forces.{index + 1}"), natoms) @ turn.T
            )
            xx, yy, zz, xy, xz, yz = table[index, 3:]
            pressure = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            virial = pressure * table[index, 2] / NKTV2P
            virials[index] = turn @ virial @ turn.T
    return energies, forces, virials


def _digest(system):
    digest = hashlib.sha256()
    for name, value in sorted(system.data.items()):
        digest.update(name.encode())
        if isinstance(value, np.ndarray):
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def _strip(source, cache, params, command, runner):
    system = dpdata.LabeledSystem(
        source, fmt="deepmd/hdf5" if "#" in source else "deepmd/npy"
    )
    if len(system) == 0:
        raise ValueError(f"Cannot strip D3 from an empty system: {source}")
    metadata = dict(
        source=source, params=params, digest=_digest(system), version=VERSION
    )
    key = hashlib.sha256(
        json.dumps([source, params], sort_keys=True).encode()
    ).hexdigest()[:24]
    destination = cache / key
    manifest = destination / "d3.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if all(previous.get(k) == v for k, v in metadata.items()):
            return destination
        dlog.info("D3: labels changed, recomputing %s", source)
        shutil.rmtree(destination)
    energies, forces, virials = compute_d3(system, params, command, runner)
    stripped = system.copy()
    stripped.data["energies"] = system["energies"] - energies
    stripped.data["forces"] = system["forces"] - forces
    if "virials" in system.data:
        stripped.data["virials"] = system["virials"] - virials
    metadata.update(
        frames=len(system),
        mean_d3_energy_per_atom=float(np.mean(energies) / system.get_natoms()),
        max_d3_force=float(np.abs(forces).max()),
    )
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        output = Path(temporary) / "system"
        stripped.to_deepmd_npy(str(output))
        (output / "d3.json").write_text(json.dumps(metadata, indent=2))
        output.rename(destination)
    dlog.info(
        "D3: stripped %d frames of %s (mean %.4f eV/atom, max |F| %.4f eV/Å)",
        metadata["frames"],
        source,
        metadata["mean_d3_energy_per_atom"],
        metadata["max_d3_force"],
    )
    return destination


def _resolve(base, source):
    file, separator, group = source.partition("#")
    return str((base / file).resolve()) + (separator + group if separator else "")


def make_d3_stripped(input_files, work_path, jdata, mdata, runner=_run_lammps):
    """Point every training input at D3-stripped copies of its systems."""
    params = strip_options(jdata)
    if params is None or not input_files:
        return
    command = mdata.get("d3_command", "lmp")
    work = Path(work_path).resolve()
    cache = Path.cwd() / CACHE
    first = Path(input_files[0]).resolve()
    template = json.loads(first.read_text())
    if "training_data" not in template["training"]:
        raise ValueError(
            "training_strip_d3 requires DeePMD-kit 2.x or newer training_data"
        )
    mapping = {}

    def stripped_paths(sources):
        paths = []
        for source in sources:
            absolute = _resolve(first.parent, source)
            if absolute not in mapping:
                cached = _strip(absolute, cache, params, command, runner)
                destination = work / WORK_DIR / cached.name
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(cached, destination)
                mapping[absolute] = f"../{WORK_DIR}/{cached.name}"
            paths.append(mapping[absolute])
        return paths

    training = stripped_paths(template["training"]["training_data"]["systems"])
    validation_data = template["training"].get("validation_data")
    validation = (
        stripped_paths(validation_data["systems"])
        if validation_data and validation_data.get("systems")
        else None
    )
    for filename in input_files:
        path = Path(filename)
        data = json.loads(path.read_text())
        data["training"]["training_data"]["systems"] = training
        if validation is not None:
            data["training"]["validation_data"]["systems"] = validation
        path.write_text(json.dumps(data, indent=4))
    dlog.info(
        "D3 stripped (%s/%s): %d systems; shared by %d models",
        params["damping_function"],
        params["functional"],
        len(mapping),
        len(input_files),
    )
