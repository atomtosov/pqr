from typing import Iterable

import numpy as np

from .singlefactor import SingleFactor


class WeightingFactor(SingleFactor):
    def weigh(self, data: np.ndarray) -> np.ndarray:
        """
        Принимает на вход данные и возвращает те же данные, но взвешенные по
        своим значениям
        :param data:
        :return:
        """
        values = self.transform(1, 0)
        if self.bigger_better:
            weights = values * data
            return weights / np.nansum(weights, axis=1)[:, np.newaxis]
        else:
            raise NotImplementedError


class EqualWeights(WeightingFactor):
    def __init__(self, shape: Iterable[int]):
        super().__init__(np.ones(shape))
