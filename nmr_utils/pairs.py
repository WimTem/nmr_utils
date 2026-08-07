from ase.io import read
from ase.geometry import get_distances
import numpy as np
from itertools import combinations


def find_pairs(atoms, cutoff=5.0, element1=None, element2=None):
    """
    Find all unique atom pairs within the cutoff.

    If element1 and element2 are given, only those pairs are returned.
    """

    distances = atoms.get_all_distances(mic=True)

    pairs = []

    for i, j in combinations(range(len(atoms)), 2):

        if distances[i, j] > cutoff:
            continue

        s1 = atoms[i].symbol
        s2 = atoms[j].symbol

        if element1 is not None and element2 is not None:

            if not (
                (s1 == element1 and s2 == element2)
                or
                (s1 == element2 and s2 == element1)
            ):
                continue

        pairs.append((i, j))

    return pairs




def find_closest_ch_pairs(structure_path, cutoff=5.0, element1="H", element2="C", n_closest=2, verbose=True):
    """
    For element1 in a structure, find the n closest element2 atoms
    within a given cutoff distance (respects periodic boundary conditions).

    Parameters
    ----------
    structure_path : str
        Path to the structure file (.cif, .pdb, CONTCAR, etc. - ASE auto-detects format).
    cutoff : float, optional
        Maximum distance (Å) to consider a H atom as "close". Default 5.0.
    element1 : str
        Default "H"
    element2 : str
        Default: "C"
    n_closest : int, optional
        Number of closest element2 atoms to report per element1 atom. Default 2.
    verbose : bool, optional
        If True, print a human-readable report. Default True.

    Returns
    -------
    dict
        Mapping {element1_atom_index: [(element2_atom_index, distance), ...]},
        sorted by increasing distance, limited to n_closest entries.

    Raises
    ------
    ValueError
        If no atoms of element1 or element2 are found in the structure.
    """
    # --- Load structure ---
    atoms = read(structure_path)

    # --- Identify atom indices by element ---
    symbols = atoms.get_chemical_symbols()
    el1_indices = [i for i, s in enumerate(symbols) if s == element1]
    el2_indices = [i for i, s in enumerate(symbols) if s == element2]

    if not el1_indices:
        raise ValueError(f"No {element1} atoms found in structure.")
    if not el2_indices:
        raise ValueError(f"No {element2} atoms found in structure.")

    # --- Compute all el1-el2 distances at once (respects PBC if atoms.cell is set) ---
    el1_positions = atoms.positions[el1_indices]
    el2_positions = atoms.positions[el2_indices]

    # get_distances handles periodic wrapping if cell/pbc are defined on `atoms`
    _, dist_matrix = get_distances(
        el1_positions, el2_positions,
        cell=atoms.get_cell(), pbc=atoms.get_pbc()
    )
    # dist_matrix shape: (n_element1, n_element2)

    # --- For each carbon, find the closest H atoms within cutoff ---
    results = {}
    for el2i, el2_idx in enumerate(el2_indices):
        dists = dist_matrix[el2i]
        order = np.argsort(dists)  # sort element1 atoms by distance
        closest = []
        for el1i in order:
            if dists[el1i] <= cutoff:
                closest.append((el1_indices[el1i], dists[el1i]))
            if len(closest) == n_closest:
                break
        results[el2_idx] = closest

    # --- Report ---
    if verbose:
        for el2_idx, el1_list in results.items():
            print(f"C{el2_idx} (atom index {el2_idx}):")
            if not el1_list:
                print(f"  No {element1} atoms within cutoff.")
            for el1_idx, d in hlist:
                print(f"  {element1}{el1_idx} (atom index {el1_idx}): {d:.3f} Å")

    return results


# Example usage:
if __name__ == "__main__":
    results = find_closest_ch_pairs('vasp_nmr/1/CONTCAR', cutoff=5.0, n_closest=3)