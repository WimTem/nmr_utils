import os
import numpy as np

from .average import average_pair
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
    print("=============================")
    print(f"Frames used: {results['nframes']}")

    spin_numbers = results["spin_numbers"]
    atom_indices = results["atom_indices"]
    nuclei = results["nuclei"]

    print("\nSpins")
    print("-----")
    for s, a, nuc in zip(spin_numbers, atom_indices, nuclei):
        print(f"  spin {s}: atom {a} ({nuc})")

    print("\nChemical shifts")
    print("---------------")
    for s in spin_numbers:
        shift = results["shifts"][s]
        # Adjust unpacking to match whatever tensor_to_shift_params returns,
        # e.g. (iso, aniso, eta) -- shown generically here:
        print(f"  spin {s}: {shift}")

    print("\nDipolar couplings")
    print("-----------------")
    for (si, sj), D_params in results["dipole_params"].items():
        D, alpha, beta, gamma = D_params
        print(f"  spins {si}-{sj}: D = {D:.2f} Hz, "
              f"(alpha, beta, gamma) = ({alpha:.1f}, {beta:.1f}, {gamma:.1f})")


def write_diagnostics(path, results):
    with open(path, "w") as f:
        f.write("=============================\n")
        f.write(f"Frames used: {results['nframes']}\n")

        f.write("\nSpins\n-----\n")
        for s, a, nuc in zip(
            results["spin_numbers"], results["atom_indices"], results["nuclei"]
        ):
            f.write(f"  spin {s}: atom {a} ({nuc})\n")

        f.write("\nChemical shifts\n---------------\n")
        for s in results["spin_numbers"]:
            f.write(f"  spin {s}: {results['shifts'][s]}\n")

        f.write("\nDipolar couplings\n-----------------\n")
        for (si, sj), D_params in results["dipole_params"].items():
            D, alpha, beta, gamma = D_params
            f.write(
                f"  spins {si}-{sj}: D = {D:.2f} Hz, "
                f"(alpha, beta, gamma) = ({alpha:.1f}, {beta:.1f}, {gamma:.1f})\n"
            )


def build_simpson_system(
    root,
    atom_i,
    atom_j,
    sigma_refs,
    nuclei=("1H", "13C"),
    output_dir="simpson_output",
    verbose=1,
):
    os.makedirs(output_dir, exist_ok=True)
    spinsys_file = os.path.join(output_dir, "spinsys.in")

    verbose_print(
        1, verbose, "Averaging dipolar and chemical shift tensors..."
    )
    (
        D_average,
        sigma_i,
        sigma_j,
        nframes,
        D_history,
        hist_i,
        hist_j,
    ) = average_pair(root, atom_i, atom_j)
    verbose_print(1, verbose, f"Averaging completed ({nframes} frames)")

    D_params = tensor_to_dipole_params(D_average)

    delta_i = shielding_to_shift(sigma_i, sigma_refs[atom_i])
    delta_j = shielding_to_shift(sigma_j, sigma_refs[atom_j])

    shift_i = tensor_to_shift_params(delta_i)
    shift_j = tensor_to_shift_params(delta_j)
    verbose_print(1, verbose, "CSA averaging completed")

    verbose_print(1, verbose, f"Writing SIMPSON file: {spinsys_file}")
    write_spinsys(
        spinsys_file,
        nuclei=list(nuclei),
        shifts={1: shift_i, 2: shift_j},
        dipoles=[(1, 2, D_params[0], D_params[1], D_params[2], D_params[3])],
    )

    results = {
        "D_average": D_average,
        "D_params": D_params,
        "delta_i": delta_i,
        "delta_j": delta_j,
        "shift_i": shift_i,
        "shift_j": shift_j,
        "nframes": nframes,
        "D_history": D_history,
        "shift_i_history": hist_i,
        "shift_j_history": hist_j,
    }

    np.savez(
        os.path.join(output_dir, "tensors.npz"),
        D_average=D_average,
        delta_i=delta_i,
        delta_j=delta_j,
    )

    write_diagnostics(os.path.join(output_dir, "diagnostics.txt"), results)

    if verbose >= 1:
        print_diagnostics(results, verbose)

    return results
