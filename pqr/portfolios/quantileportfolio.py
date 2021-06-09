from typing import Union, Tuple, Iterable
from collections import namedtuple

import numpy as np

from .portfolio import Portfolio


Quantiles = namedtuple('Quantiles', ('lower', 'upper'))


class QuantilePortfolio(Portfolio):
    _quantiles: Quantiles

    def __init__(
            self,
            quantiles: Tuple[float, float],
            budget: Union[int, float] = None,
            fee_rate: Union[int, float] = None,
            fee_fixed: Union[int, float] = None
    ):
        super().__init__(
            budget,
            fee_rate,
            fee_fixed
        )

        self.quantiles = quantiles

    def _choose_stocks(
            self,
            factor_values: np.ndarray
    ) -> np.ndarray:
        lower_threshold = np.nanquantile(
            factor_values,
            self.quantiles.lower,
            axis=1
        )
        upper_threshold = np.nanquantile(
            factor_values,
            self.quantiles.upper,
            axis=1
        )
        return (lower_threshold[:, np.newaxis] <= factor_values) & \
               (factor_values < upper_threshold[:, np.newaxis])

    @property
    def quantiles(self) -> Quantiles:
        return self._quantiles

    @quantiles.setter
    def quantiles(self, value: Tuple[float, float]) -> None:
        if isinstance(value, Iterable) \
                and len(value) == 2 \
                and all((isinstance(q, float)
                         and 0 <= q <= 1) for q in value) \
                and value[0] < value[1]:
            self._quantiles = Quantiles(*value)
        else:
            raise ValueError('quantiles must be tuple of 2 floats '
                             'in range [0, 1]')
