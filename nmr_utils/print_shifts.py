#!/usr/bin/env python
"""
Read a single .magres file and print chemical shift parameters for each atom,
using Soprano's NMR property classes.

Usage
-----
    python print_shifts_soprano.py structure.magres
    python print_shifts_soprano.py structure.magres --sigma-ref C=170.5 H=30.9
    python print_shifts_soprano.py structure.magres --atoms 0 1 2 --sigma-ref C=170.5
"""

import argparse
import numpy as np
from ase.io import read

from soprano.properties.nmr import MSIsotropy, MSAnisotropy, MSAsymmetry


def parse_sigma_ref(pairs):
    """Parse ['C=170.5', 'H=30.9'] into {'C': 170.5, 'H': 30.9}."""
    if not pairs:
        return {}
    refs = {}
    for pair in pairs:
        element, value = pair.split("=")
        refs[element.strip()] = float(value)
    return refs


def print_shifts(magres_file, atom_indices=None, sigma_refs=None):
    """
    Read a .magres file and print chemical shift parameters per atom.

    Parameters
    ----------
    magres_file : str
        Path to the .magres file.
    atom_indices : list of int, optional
        Atom indices (0-based) to print. Default: all atoms.
    sigma_refs : dict, optional
        Reference shielding per element, e.g. {"C": 170.5, "H": 30.9}.
        If omitted, raw isotropic shielding is printed instead of shift.
    """
    sigma_refs = sigma_refs or {}

    atoms = read(magres_file)

    if "ms" not in atoms.arrays:
        raise ValueError(
            "No magnetic shielding data ('ms') found in this file. "
            "Is it really a .magres file with an NMR shielding calculation?"
        )

    # Soprano computes shielding by default; passing `ref` switches it to
    # returning the chemical shift directly (shift = ref - shielding, scaled
    # by `grad`, default gradient -1.0).
    shielding = MSIsotropy.get(atoms)
    aniso = MSAnisotropy.get(atoms)
    eta = MSAsymmetry.get(atoms)

    shift = MSIsotropy.get(atoms, ref=sigma_refs) if sigma_refs else None

    indices = atom_indices if atom_indices is not None else range(len(atoms))

    np.set_printoptions(precision=4, suppress=True)

    print()
    print(f"File   : {magres_file}")
    print(f"Atoms  : {len(atoms)}")
    print()

    header = f"{'Idx':>4} {'Elem':>5} {'Shielding':>12} {'Aniso':>10} {'Eta':>8}"
    if sigma_refs:
        header += f" {'Shift':>10}"
    print(header)
    print("-" * len(header))

    for i in indices:
        element = atoms[i].symbol
        row = (
            f"{i:>4} {element:>5} {shielding[i]:>12.3f} "
            f"{aniso[i]:>10.3f} {eta[i]:>8.4f}"
        )
        if sigma_refs:
            if element in sigma_refs:
                row += f" {shift[i]:>10.3f}"
            else:
                row += f" {'--':>10}  (no ref for {element})"
        print(row)

    print()
    if not sigma_refs:
        print("Note: no --sigma-ref given, so shifts could not be computed;")
        print("only raw isotropic shielding is shown above.")


def main():
    parser = argparse.ArgumentParser(
        description="Print chemical shifts from a single .magres file using Soprano."
    )
    parser.add_argument("magres_file", help="Path to the .magres file.")
    parser.add_argument(
        "--atoms",
        nargs="+",
        type=int,
        default=None,
        help="Atom indices (0-based) to print. Default: all atoms.",
    )
    parser.add_argument(
        "--sigma-ref",
        nargs="+",
        default=None,
        metavar="ELEMENT=VALUE",
        help="Reference shielding per element, e.g. --sigma-ref C=170.5 H=30.9. "
        "If omitted, raw isotropic shielding is printed instead of shift.",
    )

    args = parser.parse_args()
    sigma_refs = parse_sigma_ref(args.sigma_ref)

    print_shifts(args.magres_file, atom_indices=args.atoms, sigma_refs=sigma_refs)


if __name__ == "__main__":
    main()