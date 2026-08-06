import numpy as np


def shielding_to_shift(shielding, sigma_ref):
    """
    Convert magnetic shielding tensor to chemical shift tensor.

    delta = sigma_ref * I - sigma

    Parameters
    ----------
    shielding : (3, 3) ndarray
        Averaged shielding tensor (ppm).
    sigma_ref : float
        Reference shielding (ppm).

    Returns
    -------
    delta : (3, 3) ndarray
        Chemical shift tensor (ppm).
    """
    return sigma_ref * np.eye(3) - shielding


def rotation_matrix_to_euler(R):
    """
    Convert a rotation matrix to ZYZ Euler angles.

    Returns radians.
    """
    beta = np.arccos(np.clip(R[2, 2], -1.0, 1.0))

    if abs(beta) < 1e-12:
        alpha = np.arctan2(R[1, 0], R[0, 0])
        gamma = 0.0
    else:
        alpha = np.arctan2(R[1, 2], R[0, 2])
        gamma = np.arctan2(R[2, 1], -R[2, 0])

    return alpha, beta, gamma


def tensor_to_shift_params(tensor):
    """
    Convert chemical shift tensor to SIMPSON parameters.

    Eigenvalues are sorted in ascending order (dxx <= dyy <= dzz).

    Returns
    -------
    iso
        Isotropic shift (ppm)
    aniso
        Shift anisotropy (ppm)
    eta
        Asymmetry parameter
    alpha, beta, gamma
        Euler angles (radians)
    """
    values, vectors = np.linalg.eigh(tensor)

    # Sort ascending: smallest -> xx, middle -> yy, largest -> zz
    idx = np.argsort(values)
    values = values[idx]
    vectors = vectors[:, idx]

    dxx, dyy, dzz = values
    iso = (dxx + dyy + dzz) / 3.0
    aniso = dzz - iso
    eta = (dyy - dxx) / aniso if abs(aniso) > 1e-12 else 0.0

    alpha, beta, gamma = rotation_matrix_to_euler(vectors)

    return iso, aniso, eta, alpha, beta, gamma


def _dominant_eigenvalue_index(values):
    """Index of the largest-magnitude eigenvalue (identifies Dzz)."""
    return np.argmax(np.abs(values))


def dipolar_from_tensor(D):
    """
    Extract dipolar coupling from averaged dipolar tensor.

    Returns Hz.
    """
    values = np.linalg.eigvalsh(D)
    Dzz = values[_dominant_eigenvalue_index(values)]
    return Dzz / 2.0


def tensor_to_dipole_params(tensor):
    """
    Convert averaged dipolar tensor to SIMPSON parameters.

    Returns
    -------
    D
        Signed dipolar coupling (Hz)
    alpha, beta, gamma
        PAS Euler angles (radians)
    """
    values, vectors = np.linalg.eigh(tensor)

    z_index = _dominant_eigenvalue_index(values)
    D = values[z_index] / 2.0

    z = vectors[:, z_index]
    xy = [i for i in range(3) if i != z_index]
    x, y = vectors[:, xy[0]], vectors[:, xy[1]]

    R = np.column_stack((x, y, z))
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1

    alpha, beta, gamma = rotation_matrix_to_euler(R)

    return D, alpha, beta, gamma


def make_spinsys(nuclei, shifts=None, dipoles=None):
    """
    Generate a SIMPSON spinsys block as text.

    Parameters
    ----------
    nuclei : list
        Example ["1H", "13C"]
    shifts : dict
        spin -> (iso, aniso, eta, alpha, beta, gamma)
    dipoles : list
        (spin1, spin2, D, alpha, beta, gamma)
    """
    channels = list(dict.fromkeys(nuclei))

    lines = [
        "spinsys {",
        "    channels " + " ".join(channels),
        "    nuclei " + " ".join(nuclei),
    ]

    if shifts is not None:
        for spin, (iso, aniso, eta, alpha, beta, gamma) in shifts.items():
            lines.append(
                f"    shift {spin} "
                f"{iso:.6f}p {aniso:.6f}p {eta:.6f} "
                f"{np.degrees(alpha):.6f} "
                f"{np.degrees(beta):.6f} "
                f"{np.degrees(gamma):.6f}"
            )

    if dipoles is not None:
        for item in dipoles:
            if len(item) == 3:
                # Fallback support for simple (i, j, D) declarations
                i, j, D = item
                lines.append(f"    dipole {i} {j} {D:.6f}")
            elif len(item) == 6:
                # Full parameter declaration with Euler angles
                i, j, D, alpha, beta, gamma = item
                lines.append(
                    f"    dipole {i} {j} {D:.6f} "
                    f"{np.degrees(alpha):.6f} "
                    f"{np.degrees(beta):.6f} "
                    f"{np.degrees(gamma):.6f}"
                )

    lines.append("}")
    return "\n".join(lines)


def write_spinsys(filename, nuclei, shifts=None, dipoles=None):
    """
    Write SIMPSON spinsys file.
    """
    text = make_spinsys(nuclei=nuclei, shifts=shifts, dipoles=dipoles)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)