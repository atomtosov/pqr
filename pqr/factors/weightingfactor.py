from typing import Iterable

import numpy as np

from .singlefactor import SingleFactor


class WeightingFactor(SingleFactor):
    """
    Class for weighting factor: factor which used to weigh positions by factor
    values
    Extends SingleFactor

    Attributes:
        dynamic: bool - is factor dynamic or not, this information is needed
        for future transformation of factor data
        bigger_better: bool | None - is better, when factor value bigger
        (e.g. ROA) or when factor value lower (e.g. P/E); if value is None it
        means that it cannot be said exactly, what is better (used for multi-
        factors)
        periodicity: DataPeriodicity - info about periodicity or discreteness
        of factor data, used for annualization and smth more
        name: str - name of factor

    Methods:
        transform() - returns transformed values of factor data
        with looking_period and lag_period (NOTE: if factor is dynamic,
        real lag = lag_period + 1)

        weigh() - weigh values in given dataset by factor values
    """

    def weigh(self, data: np.ndarray) -> np.ndarray:
        """
        Weigh values in given dataset by factor values

        :param data: given dataset (expected positions, but may be smth other)
        :return: 2-dimensional weighted matrix
        """
        values = self.transform(looking_period=1, lag_period=0)
        if self.bigger_better:
            weights = values * data
            return weights / np.nansum(weights, axis=1)[:, np.newaxis]
        else:
            raise NotImplementedError('lower_better factors '
                                      'are not supported yet')


class EqualWeights(WeightingFactor):
    """
    Simple dummy for WeightingFactor
    Inherits from WeightingFactor to provide factor, which weigh equally
    """

    def __init__(self, shape: Iterable[int]):
        """
        Initialization of EqualWeights

        :param shape: shape of dataset to be weighted (actually not)
        """
        super().__init__(np.ones(shape))
