from typing import Callable

import numpy as np

__all__ = [
    "AggregationFunction",

    "static",
    "dynamic",
]

AggregationFunction = Callable[[np.ndarray], np.ndarray]


def static(values: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(lambda x: x[0], 0, values)


def dynamic(values: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(lambda x: x[-1] / x[0] - 1, 0, values)
