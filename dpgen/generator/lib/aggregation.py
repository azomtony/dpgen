"""Bias two molecular cores (e.g. asphaltene aromatic cores) towards each other.

Opt in per ``model_devi_jobs`` entry with ``"aggregation_bias"``. Each LAMMPS
exploration task then gets:

- ``colvars.inp``: a Colvars ``distance`` between the centers of mass of the
  two cores, pulled by a moving harmonic restraint from the starting distance
  to ``target`` over ``pull_steps``, then held at ``target``.
- ``fix recenter`` (optional, on by default): the whole system is translated
  every step so the pair's center of mass sits at the box center. This adds
  no force, and keeps the pair away from the periodic boundaries.
- A task-local ``conf.lmp`` with image flags, so each core is whole and the
  second core is the periodic image nearest the first. Colvars and ``fix
  recenter`` both work with unwrapped coordinates.

Cores are either given as atom indices (0-based, POSCAR order, which is the
LAMMPS atom order) or detected: molecules from covalent bonds, then in each
molecule the largest fused ring system of sp2-like atoms (ring atoms with at
most three neighbors). Molecules whose core has at least ``min_core_atoms``
atoms count as cores; exactly two are required.

Frame selection is unchanged: frames are still picked by model deviation.
"""

import math
import os
from collections import defaultdict

import dpdata
import numpy as np

COLVARS_INPUT = "colvars.inp"
COLVARS_PREFIX = "colvars"
COLVARS_TRAJ = f"{COLVARS_PREFIX}.colvars.traj"
COVALENT_RADII = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}
KEYS = {
    "cores",
    "target",
    "k",
    "pull_steps",
    "recenter",
    "upper_wall",
    "min_core_atoms",
    "bond_scale",
}


def _positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"aggregation_bias.{name} must be a positive number")
    return float(value)


def options(job, nsteps):
    """Validated settings for one model_devi job, or None if not requested."""
    block = job.get("aggregation_bias")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError("aggregation_bias must be a dict")
    unknown = set(block) - KEYS
    if unknown:
        raise ValueError(f"Unknown aggregation_bias settings: {sorted(unknown)}")
    pull_steps = block.get("pull_steps", nsteps // 2)
    if (
        isinstance(pull_steps, bool)
        or not isinstance(pull_steps, int)
        or not 0 < pull_steps <= nsteps
    ):
        raise ValueError(
            "aggregation_bias.pull_steps must be an integer in (0, nsteps]"
        )
    config = {
        "cores": block.get("cores", "auto"),
        "target": _positive(block.get("target", 3.5), "target"),
        "k": _positive(block.get("k", 1.0), "k"),
        "pull_steps": pull_steps,
        "recenter": block.get("recenter", True),
        "upper_wall": block.get("upper_wall"),
        "min_core_atoms": block.get("min_core_atoms", 10),
        "bond_scale": _positive(block.get("bond_scale", 1.2), "bond_scale"),
    }
    if not isinstance(config["recenter"], bool):
        raise ValueError("aggregation_bias.recenter must be true or false")
    if config["upper_wall"] is not None:
        config["upper_wall"] = _positive(config["upper_wall"], "upper_wall")
        if config["upper_wall"] <= config["target"]:
            raise ValueError("aggregation_bias.upper_wall must exceed target")
    if (
        isinstance(config["min_core_atoms"], bool)
        or not isinstance(config["min_core_atoms"], int)
        or config["min_core_atoms"] < 3
    ):
        raise ValueError("aggregation_bias.min_core_atoms must be an integer >= 3")
    if config["cores"] != "auto" and not isinstance(config["cores"], (list, dict)):
        raise ValueError(
            'aggregation_bias.cores must be "auto", a pair of index lists, '
            "or a dict of pairs keyed by system index"
        )
    return config


def check_compatible(jdata, job):
    """Reject setups the bias does not support, before any task is written."""
    if jdata.get("model_devi_engine", "lammps") != "lammps":
        raise ValueError("aggregation_bias requires LAMMPS exploration")
    if "template" in job:
        raise ValueError("aggregation_bias is not supported with input templates")
    if job.get("ensemble") == "plasma" or job.get("nbeads") is not None:
        raise ValueError("aggregation_bias does not support plasma or PIMD jobs")
    if jdata.get("model_devi_nopbc", False):
        raise ValueError("aggregation_bias needs a periodic box")
    if (
        jdata.get("shuffle_poscar", False)
        and job["aggregation_bias"].get("cores", "auto") != "auto"
    ):
        raise ValueError(
            "aggregation_bias core indices do not survive shuffle_poscar; "
            'use "cores": "auto"'
        )


def _minimum_image(delta_frac):
    return delta_frac - np.round(delta_frac)


def bond_graph(coords, cell, names, types, scale):
    """Covalent bonds under periodic boundaries, as adjacency sets."""
    missing = sorted({names[t] for t in set(types)} - COVALENT_RADII.keys())
    if missing:
        raise ValueError(
            f"No covalent radius for {missing}; give aggregation_bias.cores explicitly"
        )
    radii = np.array([COVALENT_RADII[names[t]] for t in types])
    frac = coords @ np.linalg.inv(cell)
    natoms = len(coords)
    adjacency = [set() for _ in range(natoms)]
    chunk = 256
    for start in range(0, natoms, chunk):
        stop = min(start + chunk, natoms)
        delta = _minimum_image(frac[None, :, :] - frac[start:stop, None, :]) @ cell
        distance = np.linalg.norm(delta, axis=-1)
        limit = scale * (radii[start:stop, None] + radii[None, :])
        for i, j in zip(*np.nonzero((distance < limit) & (distance > 0.1))):
            adjacency[start + i].add(int(j))
            adjacency[int(j)].add(start + i)
    return adjacency


def _components(nodes, neighbors):
    seen, groups = set(), []
    for node in nodes:
        if node in seen:
            continue
        stack, group = [node], []
        seen.add(node)
        while stack:
            current = stack.pop()
            group.append(current)
            for other in neighbors(current):
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        groups.append(sorted(group))
    return groups


def _bridges(adjacency):
    """Edges on no cycle (iterative Tarjan)."""
    natoms = len(adjacency)
    order = [-1] * natoms
    low = [0] * natoms
    bridges = set()
    counter = 0
    for root in range(natoms):
        if order[root] >= 0:
            continue
        order[root] = low[root] = counter
        counter += 1
        stack = [(root, -1, iter(sorted(adjacency[root])))]
        while stack:
            node, parent, children = stack[-1]
            advanced = False
            for child in children:
                if child == parent:
                    continue
                if order[child] < 0:
                    order[child] = low[child] = counter
                    counter += 1
                    stack.append((child, node, iter(sorted(adjacency[child]))))
                    advanced = True
                    break
                low[node] = min(low[node], order[child])
            if advanced:
                continue
            stack.pop()
            if parent >= 0:
                low[parent] = min(low[parent], low[node])
                if low[node] > order[parent]:
                    bridges.add(frozenset((parent, node)))
    return bridges


def detect_cores(coords, cell, names, types, min_core_atoms=10, scale=1.2):
    """Return the fused sp2 ring core of each molecule large enough to count."""
    adjacency = bond_graph(coords, cell, names, types, scale)
    bridges = _bridges(adjacency)

    def ring_neighbors(atom):
        return [
            other
            for other in adjacency[atom]
            if frozenset((atom, other)) not in bridges
        ]

    candidates = {
        atom
        for atom in range(len(coords))
        if names[types[atom]] != "H"
        and len(adjacency[atom]) <= 3
        and ring_neighbors(atom)
    }
    molecules = _components(range(len(coords)), lambda atom: adjacency[atom])
    cores = []
    for molecule in molecules:
        members = [atom for atom in molecule if atom in candidates]
        systems = _components(
            members,
            lambda atom: [o for o in ring_neighbors(atom) if o in candidates],
        )
        if systems:
            largest = max(systems, key=len)
            if len(largest) >= min_core_atoms:
                cores.append(largest)
    return cores


def _unwrap(frac, cores):
    """Wrap all atoms into the box; pick images that keep cores whole and close.

    Returns wrapped fractional coordinates and integer image flags.
    """
    wrapped = frac - np.floor(frac)
    images = np.zeros(frac.shape, dtype=int)
    centers = []
    for core in cores:
        anchor = wrapped[core[0]]
        shift = -np.round(wrapped[core] - anchor).astype(int)
        images[core] = shift
        centers.append((wrapped[core] + shift).mean(axis=0))
    # Move the second core to the image nearest the first.
    offset = np.round(centers[1] - centers[0]).astype(int)
    images[cores[1]] -= offset
    return wrapped, images


def _ranges(ids):
    """LAMMPS group syntax with ranges: 1:4 7 9:10."""
    parts, start = [], None
    for index, value in enumerate(ids):
        if start is None:
            start = value
        if index + 1 == len(ids) or ids[index + 1] != value + 1:
            parts.append(str(start) if start == value else f"{start}:{value}")
            start = None
    return " ".join(parts)


def write_conf(path, cell, wrapped, images, types, ntypes):
    coords = wrapped @ cell
    lines = [
        "conf.lmp with image flags written by dpgen aggregation_bias",
        "",
        f"{len(types)} atoms",
        f"{ntypes} atom types",
        "",
        f"0 {cell[0][0]:.10f} xlo xhi",
        f"0 {cell[1][1]:.10f} ylo yhi",
        f"0 {cell[2][2]:.10f} zlo zhi",
        f"{cell[1][0]:.10f} {cell[2][0]:.10f} {cell[2][1]:.10f} xy xz yz",
        "",
        "Atoms # atomic",
        "",
    ]
    for index, (kind, xyz, image) in enumerate(zip(types, coords, images), 1):
        lines.append(
            f"{index} {kind + 1} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f} "
            f"{image[0]} {image[1]} {image[2]}"
        )
    with open(path, "w") as stream:
        stream.write("\n".join(lines) + "\n")


def colvars_input(cores, start, config, trj_freq):
    groups = "\n".join(
        f"    group{n} {{\n      atomNumbers {' '.join(str(i + 1) for i in core)}\n    }}"
        for n, core in enumerate(cores, 1)
    )
    text = f"""colvarsTrajFrequency {trj_freq}

colvar {{
  name core_distance
  distance {{
{groups}
  }}
}}

harmonic {{
  name pull
  colvars core_distance
  forceConstant {config["k"]:.10g}
  centers {start:.10g}
  targetCenters {config["target"]:.10g}
  targetNumSteps {config["pull_steps"]}
}}
"""
    if config["upper_wall"] is not None:
        text += f"""
harmonicWalls {{
  name keep_close
  colvars core_distance
  upperWalls {config["upper_wall"]:.10g}
  forceConstant {config["k"]:.10g}
}}
"""
    return text


def lammps_block(cores, config):
    ids = sorted(i + 1 for core in cores for i in core)
    lines = [
        "",
        "# aggregation_bias: pull the two cores together (Colvars)",
        f"group           dpgen_cores id {_ranges(ids)}",
    ]
    if config["recenter"]:
        lines.append(
            "fix             dpgen_center dpgen_cores recenter 0.5 0.5 0.5 "
            "units fraction shift all"
        )
    # The fix ID must stay fixed: LAMMPS restart files carry the Colvars state.
    lines.append(
        f"fix             dpgen_bias all colvars {COLVARS_INPUT} "
        f"output {COLVARS_PREFIX}"
    )
    return "\n".join(lines) + "\n"


def _given_cores(config, system_index):
    cores = config["cores"]
    if isinstance(cores, dict):
        for key in (system_index, str(system_index), f"{system_index:03d}"):
            if key in cores:
                return cores[key]
        raise ValueError(
            f"aggregation_bias.cores has no entry for system {system_index}"
        )
    return cores


def apply(lmp_text, conf_path, config, jdata, system_index, trj_freq):
    """Add the bias to one task. Returns the new input text and a record.

    Writes the task-local ``conf.lmp`` (replacing a link to the shared one)
    and ``colvars.inp`` next to ``conf_path``.
    """
    system = dpdata.System(conf_path, fmt="lammps/lmp", type_map=jdata["type_map"])
    cell = system["cells"][0]
    coords = system["coords"][0]
    types = system["atom_types"]
    names = system["atom_names"]
    if config["cores"] == "auto":
        cores = detect_cores(
            coords,
            cell,
            names,
            types,
            config["min_core_atoms"],
            config["bond_scale"],
        )
        if len(cores) != 2:
            raise ValueError(
                f"aggregation_bias found {len(cores)} cores with at least "
                f"{config['min_core_atoms']} atoms in {conf_path} "
                f"(sizes: {[len(c) for c in cores]}); exactly 2 are needed. "
                "Adjust min_core_atoms or give cores explicitly."
            )
    else:
        cores = _given_cores(config, system_index)
        if len(cores) != 2 or any(not core for core in cores):
            raise ValueError("aggregation_bias.cores must be two non-empty index lists")
        flat = [i for core in cores for i in core]
        if len(set(flat)) != len(flat) or any(
            isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(types)
            for i in flat
        ):
            raise ValueError(
                "aggregation_bias.cores must hold distinct 0-based atom indices"
            )
        cores = [sorted(core) for core in cores]
    frac = coords @ np.linalg.inv(cell)
    wrapped, images = _unwrap(frac, cores)
    unwrapped = (wrapped + images) @ cell
    centers = [unwrapped[core].mean(axis=0) for core in cores]
    start = float(np.linalg.norm(centers[1] - centers[0]))
    volume = abs(np.linalg.det(cell))
    widths = [
        volume / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
        for i in range(3)
    ]
    half = min(widths) / 2
    if start >= half - 0.5:
        raise ValueError(
            f"Cores in {conf_path} start {start:.2f} Å apart, too close to half "
            f"the box ({half:.2f} Å): the pull could follow a periodic image. "
            "Use a larger box or start the cores closer."
        )
    directory = os.path.dirname(conf_path)
    if os.path.islink(conf_path):
        os.remove(conf_path)
    write_conf(conf_path, cell, wrapped, images, types, len(names))
    with open(os.path.join(directory, COLVARS_INPUT), "w") as stream:
        stream.write(colvars_input(cores, start, config, trj_freq))
    marker = "timestep"
    position = lmp_text.rfind("\n" + marker)
    if position < 0:
        raise ValueError("Cannot place aggregation_bias in the LAMMPS input")
    text = (
        lmp_text[: position + 1]
        + lammps_block(cores, config)
        + lmp_text[position + 1 :]
    )
    record = {
        "cores": [[int(i) for i in core] for core in cores],
        "core_elements": core_summary(cores, names, types),
        "start_distance": start,
        "target": config["target"],
        "pull_steps": config["pull_steps"],
        "k": config["k"],
        "recenter": config["recenter"],
        "upper_wall": config["upper_wall"],
    }
    return text, record


def core_summary(cores, names, types):
    """Element counts per core, for logs."""
    out = []
    for core in cores:
        counts = defaultdict(int)
        for atom in core:
            counts[names[types[atom]]] += 1
        out.append(dict(counts))
    return out
