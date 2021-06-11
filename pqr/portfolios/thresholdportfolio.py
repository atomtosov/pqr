from typing import Union, Tuple, Iterable
from collections import namedtuple

import numpy as np

from .portfolio import Portfolio


Thresholds = namedtuple('Thresholds', ('lower', 'upper'))


class ThresholdPortfolio(Portfolio):
    _thresholds: Thresholds

    def __init__(
            self,
            thresholds: Tuple[float, float],
            budget: Union[int, float] = None,
            fee_rate: Union[int, float] = None,
            fee_fixed: Union[int, float] = None
    ):
        super().__init__(
            budget,
            fee_rate,
            fee_fixed
        )

        self.thresholds = thresholds

    def _choose_stocks(
            self,
            factor_values: np.ndarray
    ) -> np.ndarray:
        return (self.thresholds.lower <= factor_values) & \
               (factor_values < self.thresholds.upper)

    def __repr__(self) -> str:
        return f'Portfolio{self.thresholds}'

    @property
    def thresholds(self) -> Thresholds:
        return self._thresholds

    @thresholds.setter
    def thresholds(self,
                   value: Tuple[Union[int, float], Union[int, float]]) -> None:
        if isinstance(value, Iterable) \
                and len(value) == 2 \
                and all(isinstance(q, (int, float)) for q in value) \
                and value[0] < value[1]:
            self._thresholds = Thresholds(*value)
        else:
            raise ValueError('thresholds must be tuple of 2 int '
                             'or float values')
