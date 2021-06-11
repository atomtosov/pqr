from typing import Iterable

import numpy as np

from .factor import Factor


class WeightingFactor(Factor):
    def weigh(self, positions: np.ndarray) -> np.ndarray:
        if self.bigger_better:
            weights = self.values * positions
            return weights / np.nansum(weights, axis=1)[:, np.newaxis]
        else:
            raise NotImplementedError


class EqualWeights(WeightingFactor):
    def __init__(self, shape: Iterable[int]):
        super().__init__(np.ones(shape))
