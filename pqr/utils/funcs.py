import numpy as np


def lag(matrix: np.ndarray, n: int = 1):
    assert isinstance(matrix, np.ndarray) and len(matrix.shape) == 2, \
        'matrix must be 2-dimensional numpy.ndarray'
    assert isinstance(n, int), 'n must be int'
    matrix = matrix.astype(float)
    matrix = np.roll(matrix, matrix.shape[1] * n)
    if n >= 0:
        matrix[:n] = np.nan
    else:
        matrix[n:] = np.nan
    return matrix


def pct_change(matrix: np.ndarray, n: int = 1):
    assert isinstance(matrix, np.ndarray) and len(matrix.shape) == 2, \
        'matrix must be 2-dimensional numpy.ndarray'
    assert isinstance(n, int) and n >= 0, \
        'n must be int and >= 0'
    matrix = matrix.astype(float)
    pct_changes = matrix / lag(matrix, n) - 1
    pct_changes[(pct_changes == np.inf) | (pct_changes == -np.inf)] = np.nan
    return pct_changes


def make_intervals(array: np.ndarray) -> np.ndarray:
    assert array.ndim == 1, 'array must be 1-dimensional'
    n = np.size(array) - 1
    return np.take(
        array,
        np.arange(n * 2).reshape((n, -1)) - np.indices((n, 2))[0]
    )
