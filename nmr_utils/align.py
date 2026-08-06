from ase.build.rotate import rotation_matrix_from_points
import numpy as np


def kabsch_rotation(
    mobile,
    reference,
):
    """
    Calculate optimal rotation matrix
    that aligns mobile onto reference.
    """

    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)

    X = mobile - mobile_center
    Y = reference - reference_center


    H = X.T @ Y


    U, S, Vt = np.linalg.svd(H)


    R = Vt.T @ U.T


    # Avoid improper rotation (reflection)
    if np.linalg.det(R) < 0:

        Vt[-1,:] *= -1

        R = Vt.T @ U.T


    return R



def align_to_reference(
    atoms,
    reference,
    mask=None,
):
    """
    Align atoms onto reference structure.

    Returns
    -------
    atoms
        Rotated structure.

    R
        Rotation matrix.
    """

    if mask is None:

        mask = np.ones(
            len(atoms),
            dtype=bool
        )


    mobile = atoms.positions[mask]

    target = reference.positions[mask]


    R = kabsch_rotation(
        mobile,
        target,
    )


    # translate to common origin

    atoms.positions -= (
        atoms.positions[mask].mean(axis=0)
    )


    atoms.positions = (
        atoms.positions @ R
    )


    atoms.positions += (
        target.mean(axis=0)
    )


    return atoms, R