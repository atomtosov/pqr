from typing import Iterable, Union

import numpy as np

from pqr.base.multi_factor import BaseMultiFactor
from pqr.base.factor import ChoosingFactorInterface, BaseFactor
from pqr.base.limits import BaseLimits, Quantiles
from pqr.utils import epsilon


class WeightedMultiFactor(BaseMultiFactor, ChoosingFactorInterface):
    _weights: np.ndarray

    def __init__(
            self,
            factors: Iterable[BaseFactor],
            weights: Iterable[Union[int, float]] = None
    ):
        super().__init__(factors)

        self.weights = weights

    def _calc_values(self):
        factors_values = np.array(
            [factor.values for factor in self.factors]
        )
        weighted_factor_values = factors_values * \
            self.weights[:, np.newaxis, np.newaxis]
        return np.nansum(weighted_factor_values, axis=0)

    def choose(self, data: np.ndarray, by: BaseLimits):
        if isinstance(by, Quantiles):
            values = self._calc_values()
            values[np.isnan(data)] = np.nan

            lower_threshold = np.nanquantile(values, by.lower, axis=1)
            upper_threshold = np.nanquantile(values, by.upper, axis=1)
            # to include stock with highest factor value
            if by.upper == 1:
                upper_threshold += epsilon
            choice = (lower_threshold[:, np.newaxis] <= values) & \
                     (values < upper_threshold[:, np.newaxis])
            data = (data * choice).astype(float)
            data[data == 0] = np.nan
            return ~np.isnan(data)
        elif isinstance(by, BaseLimits):
            raise NotImplementedError
        else:
            raise ValueError('by must be Limits')

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value: Union[Iterable[Union[int, float]], None]):
        if value is None:
            self._weights = np.ones(len(self.factors)) / len(self.factors)
        elif np.all([isinstance(w, (int, float)) and w > 0 for w in value]):
            # normalize weights if necessary (sum of weights must be = 1)
            value = np.array(value) / (np.sum(value)) \
                if np.sum(value) != 1 else 1
            self._weights = value
        else:
            raise ValueError('weights must be Iterable of int or float > 0')
