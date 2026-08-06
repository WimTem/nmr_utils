import argparse
import numpy as np

from .average import average_tensor


def main():

    parser = argparse.ArgumentParser(
        description="Average dipolar tensors over a trajectory."
    )

    parser.add_argument(
        "directory",
        help="Root directory containing snapshot subdirectories.",
    )

    parser.add_argument(
        "--atoms",
        nargs=2,
        type=int,
        required=True,
        metavar=("I", "J"),
        help="Atom indices.",
    )

    args = parser.parse_args()

    D = average_tensor(
        args.directory,
        args.atoms[0],
        args.atoms[1],
    )

    eigvals, eigvecs = np.linalg.eigh(D)

    np.set_printoptions(precision=6, suppress=True)

    print()
    print(f"Frames processed : {nframes}")
    print(f"Atom pair        : ({args.atoms[0]}, {args.atoms[1]})")
    print()

    print("Average dipolar tensor")
    print("----------------------")
    print(D)
    print()

    print("Principal values")
    print("----------------")
    print(eigvals)
    print()

    print("Principal axes")
    print("----------------")
    print(eigvecs)