#!/usr/bin/env python3
"""Generate DP-GEN exploration POSCAR files from seed structures."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create scaled, strained, and rattled POSCAR files from seed POSCARs. "
            "Use these files as DP-GEN sys_configs."
        )
    )
    parser.add_argument("--seed-dir", required=True, help="Directory containing seed POSCAR files.")
    parser.add_argument("--output", required=True, help="Output directory for generated POSCARs.")
    parser.add_argument(
        "--names",
        nargs="*",
        default=["POSCAR", "CONTCAR"],
        help="Seed filenames to search for recursively.",
    )
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=[1, 1, 1],
        metavar=("NX", "NY", "NZ"),
        help="Repeat each seed structure before perturbing.",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[0.98, 1.0, 1.02],
        help="Isotropic cell scale factors.",
    )
    parser.add_argument(
        "--n-pert",
        type=int,
        default=10,
        help="Number of perturbed structures per seed and scale.",
    )
    parser.add_argument(
        "--rattle",
        type=float,
        default=0.03,
        help="Gaussian position noise in Angstrom.",
    )
    parser.add_argument(
        "--strain",
        type=float,
        default=0.02,
        help="Uniform random strain magnitude applied to the cell.",
    )
    parser.add_argument("--seed", type=int, default=20260714, help="Random seed.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files.",
    )
    return parser.parse_args()


def import_ase():
    try:
        from ase.io import read, write
    except ImportError as exc:
        raise SystemExit(
            "This script needs ASE. Activate the DP-GEN/DeePMD environment, "
            "or install ASE in the environment used for preprocessing."
        ) from exc
    return read, write


def find_seed_files(seed_dir: Path, names: list[str]) -> list[Path]:
    seeds: list[Path] = []
    wanted = set(names)
    for path in seed_dir.rglob("*"):
        if path.is_file() and path.name in wanted:
            seeds.append(path)
    return sorted(seeds)


def seed_label(seed_dir: Path, seed_file: Path) -> str:
    rel = seed_file.relative_to(seed_dir)
    if seed_file.name in {"POSCAR", "CONTCAR"} and len(rel.parts) > 1:
        return "__".join(rel.parts[:-1])
    return "__".join(rel.parts).replace(".", "_")


def random_strain_matrix(rng: np.random.Generator, magnitude: float) -> np.ndarray:
    strain = rng.uniform(-magnitude, magnitude, size=(3, 3))
    strain = 0.5 * (strain + strain.T)
    matrix = np.eye(3) + strain
    if np.linalg.det(matrix) <= 0.0:
        return np.eye(3)
    return matrix


def main() -> None:
    args = parse_args()
    seed_dir = Path(args.seed_dir).expanduser().resolve()
    output = Path(args.output).expanduser()
    read, write = import_ase()

    seeds = find_seed_files(seed_dir, args.names)
    if not seeds:
        raise SystemExit(f"No seed files named {args.names} found under {seed_dir}")

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    n_written = 0

    for seed_file in seeds:
        label = seed_label(seed_dir, seed_file)
        atoms0 = read(seed_file)
        atoms0 = atoms0.repeat(tuple(args.supercell))

        for scale in args.scales:
            scale_dir = output / label / f"scale-{scale:.3f}"
            for idx in range(args.n_pert):
                atoms = atoms0.copy()
                atoms.set_cell(atoms.cell.array * scale, scale_atoms=True)
                if args.strain > 0.0:
                    atoms.set_cell(
                        atoms.cell.array @ random_strain_matrix(rng, args.strain),
                        scale_atoms=True,
                    )
                if args.rattle > 0.0:
                    atoms.positions += rng.normal(0.0, args.rattle, size=atoms.positions.shape)
                    atoms.wrap()

                task_dir = scale_dir / f"{idx:06d}"
                poscar = task_dir / "POSCAR"
                if args.dry_run:
                    print(poscar)
                else:
                    task_dir.mkdir(parents=True, exist_ok=True)
                    write(poscar, atoms, format="vasp", direct=True, vasp5=True, sort=False)
                n_written += 1

    action = "Would write" if args.dry_run else "Wrote"
    print(f"{action} {n_written} POSCAR files from {len(seeds)} seed structures.")


if __name__ == "__main__":
    main()
