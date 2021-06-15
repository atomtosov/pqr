from typing import Iterable, Sequence, Union

import numpy as np

from .multifactor import MultiFactor
from pqr.factors import Factor
from pqr.utils import Quantiles, epsilon


class WeightedMultiFactor(MultiFactor):
    _weights: np.ndarray

    def __init__(
            self,
            factors: Sequence[Factor],
            weights: Iterable[Union[int, float]] = None,
            name: str = None
    ):
        super().__init__(factors, name)

        self.weights = weights

    def transform(self,
                  looking_period: int = 1,
                  lag_period: int = 1) -> np.ndarray:
        factors = np.array(
            [factor.transform(looking_period, lag_period)
             for factor in self.factors]
        )
        weighted_factors = factors * self.weights[:, np.newaxis, np.newaxis]
        return np.nansum(weighted_factors, axis=0)

    def choose(self,
               data: np.ndarray,
               interval: Quantiles,
               looking_period: int = 1,
               lag_period: int = 0) -> np.ndarray:
        if not isinstance(interval, Quantiles):
            raise ValueError('interval must be Quantiles')

        values = self.transform(looking_period, lag_period)
        values[np.isnan(data)] = np.nan

        lower_threshold = np.nanquantile(values, interval.lower, axis=1)
        upper_threshold = np.nanquantile(values, interval.upper, axis=1)
        # to include stock with highest factor value
        if interval.upper == 1:
            upper_threshold += epsilon
        choice = (lower_threshold[:, np.newaxis] <= values) & \
                 (values < upper_threshold[:, np.newaxis])
        data = (data * choice).astype(float)
        data[data == 0] = np.nan
        return ~np.isnan(data)

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
