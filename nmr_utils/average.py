import numpy as np

from soprano.properties.nmr import DipolarCoupling

import itertools
from .io import iter_snapshots
from .pairs import find_pairs
from .align import align_to_reference


def _load_frames(root):
    """
    Load all MD snapshots from `root`, using the first frame as the
    alignment reference (shared by every averaging routine below so
    the reference is always frame 0, consistently).

    Returns
    -------
    reference : Atoms
        The first snapshot, used as the alignment reference.
    frames : list
        All snapshots, including the reference, in original order.
    """
    snapshots = iter_snapshots(root)
    reference = next(snapshots)
    frames = [reference, *snapshots]
    return reference, frames


def _dipolar_tensor(d, rhat):
    """Build the 3x3 dipolar tensor from coupling constant and unit vector."""
    return d * (3 * np.outer(rhat, rhat) - np.eye(3))


def _update_running_average(avg, value, n):
    """
    In-place incremental mean update: avg <- avg + (value - avg) / n.
    `n` is the count *including* `value` (i.e. call after incrementing
    the frame counter).
    """
    avg += (value - avg) / n
    return avg
    
    
def average_cluster(root, atom_indices, mask=None):
    """
    Single-pass averaging of all pairwise dipolar tensors and all
    per-atom shielding tensors for a cluster of spins, avoiding
    repeated trajectory reads.

    Returns
    -------
    D_averages : dict {(atom_i, atom_j): ndarray(3,3)}
    sigma_averages : dict {atom_index: ndarray(3,3)}
    nframes : int
    D_histories : dict {(atom_i, atom_j): list of dict}
    sigma_histories : dict {atom_index: list of ndarray}
    """
    reference, frames = _load_frames(root)

    pairs = list(itertools.combinations(sorted(set(atom_indices)), 2))
    unique_atoms = sorted(set(atom_indices))

    D_averages = {pair: np.zeros((3, 3)) for pair in pairs}
    sigma_averages = {a: np.zeros((3, 3)) for a in unique_atoms}

    D_histories = {pair: [] for pair in pairs}
    sigma_histories = {a: [] for a in unique_atoms}

    nframes = 0

    for atoms in frames:
        atoms, R = align_to_reference(atoms, reference, mask=mask)
        couplings = DipolarCoupling.get(atoms)
        nframes += 1

        for pair in pairs:
            d, rhat = couplings[pair]  # pair is already sorted
            D = _dipolar_tensor(d, rhat)
            _update_running_average(D_averages[pair], D, nframes)
            D_histories[pair].append({"d": d, "rhat": rhat.copy(), "D": D.copy()})

        for a in unique_atoms:
            sigma = R @ np.asarray(atoms.arrays["ms"][a]) @ R.T
            _update_running_average(sigma_averages[a], sigma, nframes)
            sigma_histories[a].append(sigma.copy())

    return D_averages, sigma_averages, nframes, D_histories, sigma_histories


def average_tensor(root, atom_i, atom_j):
    reference, frames = _load_frames(root)

    D_average = np.zeros((3, 3))
    instantaneous = []
    nframes = 0

    key = tuple(sorted((atom_i, atom_j)))  # canonical order for dict lookup

    for atoms in frames:
        atoms, R = align_to_reference(atoms, reference)

        couplings = DipolarCoupling.get(atoms)
        d, rhat = couplings[key]
        D = _dipolar_tensor(d, rhat)

        instantaneous.append({"d": d, "rhat": rhat.copy(), "D": D.copy()})

        nframes += 1
        _update_running_average(D_average, D, nframes)

    return D_average, nframes, instantaneous


def average_shift_tensor(root, atom_index, mask=None):
    """
    Dynamically average magnetic shielding tensors.
    """
    reference, frames = _load_frames(root)

    sigma_average = np.zeros((3, 3))
    instantaneous = []
    nframes = 0

    for atoms in frames:
        atoms, R = align_to_reference(atoms, reference, mask=mask)

        # ASE/Soprano magres shielding tensor
        sigma = np.asarray(atoms.arrays["ms"][atom_index])

        # Rotate tensor into reference frame
        sigma = R @ sigma @ R.T

        instantaneous.append(sigma.copy())

        nframes += 1
        _update_running_average(sigma_average, sigma, nframes)

    return sigma_average, nframes, instantaneous


def average_pair(root, atom_i, atom_j, mask=None):
    reference, frames = _load_frames(root)

    D_average = np.zeros((3, 3))
    sigma_i_average = np.zeros((3, 3))
    sigma_j_average = np.zeros((3, 3))

    D_history = []
    sigma_i_history = []
    sigma_j_history = []

    nframes = 0
    key = tuple(sorted((atom_i, atom_j)))  # <-- add this

    for atoms in frames:
        atoms, R = align_to_reference(atoms, reference, mask=mask)

        couplings = DipolarCoupling.get(atoms)
        d, rhat = couplings[key]  # <-- use key instead of (atom_i, atom_j)
        D = _dipolar_tensor(d, rhat)

        sigma_i = R @ np.asarray(atoms.arrays["ms"][atom_i]) @ R.T
        sigma_j = R @ np.asarray(atoms.arrays["ms"][atom_j]) @ R.T

        nframes += 1
        _update_running_average(D_average, D, nframes)
        _update_running_average(sigma_i_average, sigma_i, nframes)
        _update_running_average(sigma_j_average, sigma_j, nframes)

        D_history.append({"d": d, "rhat": rhat.copy(), "D": D.copy()})
        sigma_i_history.append(sigma_i.copy())
        sigma_j_history.append(sigma_j.copy())

    return (
        D_average, sigma_i_average, sigma_j_average, nframes,
        D_history, sigma_i_history, sigma_j_history,
    )


def average_all(
    root,
    cutoff=4.5,
    element1=None,
    element2=None,
    mask=None,
    store_history=False,
):
    """
    Dynamically average dipolar tensors from MD snapshots.

    Returns:
        averages
            Final averaged tensor for each pair.

        nframes
            Number of snapshots.

        history
            Running tensor averages (optional).

        instantaneous
            Raw D(t), d(t), and rhat(t) for every pair.
            Used for bootstrap and SIMPSON.
    """
    reference, frames = _load_frames(root)

    if mask is None:
        mask = np.ones(len(reference), dtype=bool)

    pairs = find_pairs(
        reference, cutoff=cutoff, element1=element1, element2=element2
    )

    averages = {pair: np.zeros((3, 3)) for pair in pairs}
    instantaneous = {pair: [] for pair in pairs}
    history = {pair: [] for pair in pairs} if store_history else None

    nframes = 0

    for atoms in frames:
        atoms, R = align_to_reference(atoms, reference, mask=mask)
        couplings = DipolarCoupling.get(atoms)
        nframes += 1

        for pair in pairs:
            d, rhat = couplings[pair]
            D = _dipolar_tensor(d, rhat)

            instantaneous[pair].append({"d": d, "rhat": rhat.copy(), "D": D.copy()})
            _update_running_average(averages[pair], D, nframes)

            if store_history:
                history[pair].append(averages[pair].copy())

    if store_history:
        return averages, nframes, history, instantaneous

    return averages, nframes, instantaneous
