import os
import numpy as np
import itertools

from .average import average_tensor, average_shift_tensor, average_cluster
from .simpson import (
    shielding_to_shift,
    tensor_to_shift_params,
    tensor_to_dipole_params,
    write_spinsys,
)
from .io import iter_snapshots


def verbose_print(level, verbose, *args):
    """Print only if requested verbosity is reached."""
    if verbose >= level:
        print(*args)


def tensor_stats(tensor):
    """Basic tensor diagnostics: eigenvalues, span, isotropic value."""
    eig = np.linalg.eigvalsh(tensor)
    return {"eig": eig, "span": eig[-1] - eig[0], "iso": np.mean(eig)}


def _shift_tensor_report(results, spin_labels=("Spin 1", "Spin 2")):
    """
    Yield (label, tensor_stats) for the two chemical-shift tensors stored
    in `results`. Shared by both the console and file diagnostics writers
    so the eigenvalue/span/iso computation isn't duplicated.
    """
    for label, key in zip(spin_labels, ("delta_i", "delta_j")):
        yield label, tensor_stats(results[key])


def print_diagnostics(results, verbose=1):
    print()
    print("mdhetcor averaging diagnostics")
    print("=============================")
    print(f"Frames used: {results['nframes']}")

    D, alpha, beta, gamma = results["D_params"]

    print("\nDipolar coupling")
    print("----------------")
    print(f"D = {D:.3f} Hz")
    print(f"PAS Euler angles (rad): {alpha:.4f}, {beta:.4f}, {gamma:.4f}")

    print("\nDipolar tensor")
    print("----------------")
    print(results["D_average"])
    print("Eigenvalues:", np.linalg.eigvalsh(results["D_average"]))

    for label, stats in _shift_tensor_report(
        results, spin_labels=("Spin 1 CSA", "Spin 2 CSA")
    ):
        print(f"\n{label}")
        print("-" * len(label))
        print("Eigenvalues:", stats["eig"])
        print("Iso:", stats["iso"], "ppm")
        print("Span:", stats["span"], "ppm")

    if verbose >= 2:
        print("\nStored trajectories")
        print("------------------")
        print("Dipolar frames:", len(results["D_history"]))
        print("Spin 1 CSA frames:", len(results["shift_i_history"]))
        print("Spin 2 CSA frames:", len(results["shift_j_history"]))


def write_diagnostics(filename, results):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("mdhetcor averaging diagnostics\n")
        f.write("=============================\n\n")
        f.write(f"Frames used: {results['nframes']}\n\n")

        f.write("Dipolar coupling\n")
        f.write("----------------\n")

        D, alpha, beta, gamma = results["D_params"]
        f.write(f"D = {D:.6f} Hz\n")
        f.write(
            f"Euler angles = {alpha:.5f} {beta:.5f} {gamma:.5f}\n\n"
        )

        f.write("Dipolar tensor eigenvalues\n")
        f.write(str(np.linalg.eigvalsh(results["D_average"])))
        f.write("\n\n")

        for label, stats in _shift_tensor_report(
            results, spin_labels=("Spin 1 shift", "Spin 2 shift")
        ):
            f.write(f"{label}\n")
            f.write("-" * len(label) + "\n")
            f.write(f"Eigenvalues: {stats['eig']}\n")
            f.write(f"Span: {stats['span']:.4f} ppm\n")
            f.write(f"Isotropic: {stats['iso']:.4f} ppm\n\n")


def build_simpson_system(
    root,
    atom_indices,
    nuclei,
    sigma_refs,
    output_dir="simpson_output",
    label=None,
    verbose=1,
):
    """
    Build a SIMPSON spin system with an arbitrary number of spins,
    using a single pass over the trajectory (via average_cluster) to
    get all dipolar and shift tensors at once.
    """
    if len(atom_indices) != len(nuclei):
        raise ValueError("atom_indices and nuclei must be the same length.")
    n_spins = len(atom_indices)
    if n_spins < 2:
        raise ValueError("Need at least 2 spins to build a spin system.")

    os.makedirs(output_dir, exist_ok=True)

    if label is None:
        label = "_".join(str(idx) for idx in atom_indices)
    spinsys_file = os.path.join(output_dir, f"{label}.in")

    spin_numbers = list(range(1, n_spins + 1))
    atom_to_spin = dict(zip(atom_indices, spin_numbers))

    # ---------------------------------------------------------------
    # Single pass: all dipolar pairs + all shielding tensors together
    # ---------------------------------------------------------------
    verbose_print(1, verbose, f"Averaging dipolar + CS tensors in one pass ({n_spins} spins)...")
    D_averages, sigma_averages, nframes, D_histories, sigma_histories = average_cluster(
        root, atom_indices
    )
    verbose_print(1, verbose, f"Averaging completed ({nframes} frames)")

    # ---------------------------------------------------------------
    # Per-spin chemical shifts (convert shielding -> shift, per atom)
    # ---------------------------------------------------------------
    shifts = {}
    shift_deltas = {}
    shift_histories = {}

    for atom_idx in atom_indices:
        spin_num = atom_to_spin[atom_idx]
        sigma = sigma_averages[atom_idx]

        if atom_idx in sigma_refs:
            delta = shielding_to_shift(sigma, sigma_refs[atom_idx])
        else:
            delta = sigma  # no reference given; assume already shift-convention

        shift_params = tensor_to_shift_params(delta)

        shifts[spin_num] = shift_params
        shift_deltas[spin_num] = delta
        # Convert per-frame shielding history to shift too, for diagnostics
        if atom_idx in sigma_refs:
            shift_histories[spin_num] = [
                shielding_to_shift(s, sigma_refs[atom_idx])
                for s in sigma_histories[atom_idx]
            ]
        else:
            shift_histories[spin_num] = sigma_histories[atom_idx]

    verbose_print(1, verbose, "CSA -> shift conversion completed for all spins")

    # ---------------------------------------------------------------
    # Pairwise dipolar params (convert tensor -> SIMPSON dipole params)
    # ---------------------------------------------------------------
    dipoles = []
    dipole_params = {}
    dipole_histories = {}

    for (atom_i, atom_j), D_avg in D_averages.items():
        si, sj = atom_to_spin[atom_i], atom_to_spin[atom_j]
        D_params = tensor_to_dipole_params(D_avg)

        dipoles.append((si, sj, D_params[0], D_params[1], D_params[2], D_params[3]))
        dipole_params[(si, sj)] = D_params
        dipole_histories[(si, sj)] = D_histories[(atom_i, atom_j)]

    verbose_print(1, verbose, f"Dipole parameter conversion completed for {len(dipoles)} pair(s)")

    # ---------------------------------------------------------------
    # Write SIMPSON spinsys file
    # ---------------------------------------------------------------
    verbose_print(1, verbose, f"Writing SIMPSON file: {spinsys_file}")
    write_spinsys(
        spinsys_file,
        nuclei=list(nuclei),
        shifts=shifts,
        dipoles=dipoles,
    )

    # ---------------------------------------------------------------
    # Collect results / diagnostics
    # ---------------------------------------------------------------
    results = {
        "atom_indices": atom_indices,
        "nuclei": nuclei,
        "spin_numbers": spin_numbers,
        "shift_deltas": shift_deltas,
        "shifts": shifts,
        "shift_histories": shift_histories,
        "dipole_params": dipole_params,
        "dipole_histories": dipole_histories,
        "nframes": nframes,
    }

    if verbose >= 1:
        print_diagnostics(results, verbose)
        write_diagnostics(os.path.join(output_dir, f"{label}_diagnostics.txt"), results)

        np.savez(
            os.path.join(output_dir, f"{label}_tensors.npz"),
            atom_indices=np.array(atom_indices),
            nuclei=np.array(nuclei),
            shift_deltas=np.array([shift_deltas[s] for s in spin_numbers]),
            dipole_pairs=np.array(list(dipole_params.keys())),
            dipole_D_avg=np.array([dipole_params[k] for k in dipole_params]),
        )

    return results

