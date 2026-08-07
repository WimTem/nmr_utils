#!/usr/bin/env python
"""
Read a single .magres file and print chemical shift parameters for each atom,
using Soprano's NMR property classes.

Usage
-----
    python print_shifts_soprano.py structure.magres
    python print_shifts_soprano.py structure.magres --sigma-ref C=170.5 H=30.9
    python print_shifts_soprano.py structure.magres --atoms 0 1 2 --sigma-ref C=170.5
    python print_shifts_soprano.py structure.magres --elements C H --sigma-ref C=170.5 H=30.9
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


def print_shifts(magres_file, atom_indices=None, elements=None, sigma_refs=None):
    """
    Read a .magres file and print chemical shift parameters per atom.


    Parameters
    ----------
    magres_file : str
        Path to the .magres file.
    atom_indices : list of int, optional
        Atom indices (0-based) to print. Default: all atoms.
    elements : list of str, optional
        Chemical element symbols to filter by, e.g. ["C", "H"]. Default: all
        elements. Combined with `atom_indices` as an intersection if both
        are given.
    sigma_refs : dict, optional
        Reference shielding per element, e.g. {"C": 170.5, "H": 30.9}.
        If omitted, raw isotropic shielding is printed instead of shift.
    """
    sigma_refs = sigma_refs or {}

    if magres_file[-6:] == 'OUTCAR':
        atoms = read_vasp(magres_file,format='vasp-out')
    else:
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

    if elements is not None:
        element_set = set(elements)
        indices = [i for i in indices if atoms[i].symbol in element_set]
        if not indices:
            print(f"No atoms found matching elements: {sorted(element_set)}")
            return

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
        "--elements",
        nargs="+",
        default=None,
        metavar="ELEMENT",
        help="Chemical elements to filter by, e.g. --elements C H. "
        "Default: all elements. Combined with --atoms as an intersection "
        "if both are given.",
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

    print_shifts(
        args.magres_file,
        atom_indices=args.atoms,
        elements=args.elements,
        sigma_refs=sigma_refs,
    )

from pathlib import Path

import ase.io
import numpy as np
import scipy.constants as const

def _find_line(lines: list[str], marker: str) -> int:
    """Return the index of the last line containing *marker*, or raise."""
    idx = None
    for i, line in enumerate(lines):
        if marker in line:
            idx = i
    if idx is None:
        raise ValueError(
            f"Expected OUTCAR section '{marker}' not found. "
            "Is this a VASP NMR calculation OUTCAR?"
        )
    return idx


def read_vasp(file: str, format: str) -> ase.Atoms:
    """
    Read NMR data from a VASP OUTCAR file.

    Parses electric field gradients (EFG) and magnetic shielding tensors from
    a VASP NMR calculation and attaches them to an ASE Atoms object as arrays
    named ``'efg'`` and ``'ms'``.

    Parameters
    ----------
    file : str
        Path to the VASP OUTCAR file.
    format : str
        ASE format string for reading the structure (e.g., ``'vasp-out'``).

    Returns
    -------
    ase.Atoms
        Atoms object with ``'efg'`` and ``'ms'`` arrays attached.

    Raises
    ------
    ValueError
        If the OUTCAR is missing expected NMR sections.

    Examples
    --------
    >>> atoms = read_vasp('OUTCAR', 'vasp-out')
    >>> atoms.get_array('ms')  # magnetic shielding tensors
    """
    atoms = ase.io.read(file, format=format)
    n_atoms = atoms.get_global_number_of_atoms()
    np.set_printoptions(suppress=True)

    with Path(file).open() as outcar:
        lines = outcar.readlines()

    # Locate required OUTCAR sections
    idx_efg = _find_line(lines, "Electric field gradients (V/A^2)")
    idx_sym = _find_line(lines, "SYMMETRIZED TENSORS")
    idx_g0 = _find_line(lines, "G=0 CONTRIBUTION TO CHEMICAL SHIFT")
    idx_core = _find_line(lines, "Core NMR properties")

    # Magnetic susceptibility
    idx_sus = _find_line(lines, "Core contribution to magnetic susceptibility:")
    sus_parts = lines[idx_sus].split()
    mag_sus = float(sus_parts[5]) * 10 ** int(sus_parts[6][-2:])

    # Cell volume (first occurrence)
    volume = None
    for line in lines:
        if "volume of cell" in line:
            volume = float(line.split()[4])
            break
    if volume is None:
        raise ValueError("Cell volume not found in OUTCAR.")

    # Avogadro-based conversion factor for magnetic susceptibility
    chi_fact = 3.0 / 8.0 / np.pi * volume * 6.022142e23 / 1e24

    # --- EFG tensors ---
    efg = []
    for i in range(n_atoms):
        # EFG data starts 4 lines after the header: V_xx, V_yy, V_zz, V_xy, V_xz, V_yz
        grad = lines[idx_efg + 4 + i].split()[1:]
        matrix = np.array(
            [
                [grad[0], grad[3], grad[4]],  # xx xy xz
                [grad[3], grad[1], grad[5]],  # xy yy yz
                [grad[4], grad[5], grad[2]],  # xz yz zz
            ]
        )
        efg.append(matrix)

    # --- Symmetrized shielding tensors (3 rows per atom, every 4th line is a separator) ---
    sym_tensor = []
    for i in range(n_atoms * 4):
        if i % 4 != 0:
            sym_tensor.append(lines[idx_sym + i + 1].split())

    # --- G=0 constant shielding (3x3 tensor) ---
    # VASP 6.4.1+ has an extra description line
    if lines[idx_g0 + 1].strip() == "using pGv susceptibility, excluding core contribution":
        start_idx = idx_g0 + 5
    else:
        start_idx = idx_g0 + 4

    const_shield = []
    for i in range(3):
        const_shield.append(lines[start_idx + i].split()[1:])

    # --- Core shielding per element type ---
    unique_elements = np.unique(atoms.get_chemical_symbols())
    core_shield_rows = []
    for i in range(len(unique_elements)):
        core_shield_rows.append(lines[idx_core + 4 + i].split()[1:])
    core_shield_map = {row[0]: float(row[1]) for row in core_shield_rows}
    core_shield = np.array([core_shield_map[el] for el in atoms.get_chemical_symbols()])

    # --- Assemble results ---
    efg = np.array(efg, dtype=float)
    efg = efg * 1e20 / const.physical_constants["atomic unit of electric field gradient"][0]
    sym_tensor = np.split(np.array(sym_tensor, dtype=float), n_atoms)
    const_shield = np.array(const_shield, dtype=float)

    ms = []
    for i in range(n_atoms):
        core_diag = np.diag(core_shield[i] * np.ones(3))
        ms_tensor = (
            sym_tensor[i]
            + const_shield
            + core_diag
            + mag_sus / chi_fact * 1e6 * np.eye(3)
        )
        ms.append(-ms_tensor)
    ms = np.array(np.array(ms).tolist(), dtype=float)

    atoms.set_array('efg', efg)
    atoms.set_array('ms', ms)

    return atoms


if __name__ == "__main__":
    main()