import numpy as np


def make_intervals(array: np.ndarray) -> np.ndarray:
    assert array.ndim == 1, 'array must be 1-dimensional'
    n = np.size(array) - 1
    return np.take(
        array,
        np.arange(n * 2).reshape((n, -1)) - np.indices((n, 2))[0]
    )
