import numpy as np
import numpy.typing as npt

__all__ = [
    "static",
    "dynamic",
]


def static(values: np.ndarray) -> npt.ArrayLike:
    if values.ndim == 2:
        return np.apply_along_axis(lambda x: x[0], 0, values)
    return values[0]


def dynamic(values: np.ndarray) -> npt.ArrayLike:
    if values.ndim == 2:
        return np.apply_along_axis(lambda x: x[-1] / x[0] - 1, 0, values)
    return values[-1] / values[0] - 1
