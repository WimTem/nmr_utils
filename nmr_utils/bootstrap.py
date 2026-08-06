import numpy as np


def bootstrap_tensor(frame_tensors, n_boot=5000):
    """
    Bootstrap averaged dipolar tensor.

    Parameters
    ----------
    frame_tensors : list
        List of instantaneous 3x3 tensors.
    n_boot : int
        Number of bootstrap replicas.

    Returns
    -------
    boot_tensors : ndarray
        Shape (n_boot, 3, 3).
    """
    tensors = np.asarray(frame_tensors)
    nframes = len(tensors)

    indices = np.random.randint(0, nframes, size=(n_boot, nframes))
    return tensors[indices].mean(axis=1)