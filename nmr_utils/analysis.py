import numpy as np


def running_tensor_average(history):
    """
    Calculate cumulative tensor averages.

    Returns
    -------
    averages : list of tensors
        averages[i] is the mean of history[0:i+1].
    """
    averages = []
    avg = np.zeros_like(history[0])
    for i, T in enumerate(history, start=1):
        avg += (T - avg) / i
        averages.append(avg.copy())
    return averages