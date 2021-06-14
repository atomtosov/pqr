from abc import ABC
from typing import Union

import numpy as np


class BaseLimits(ABC):
    def __init__(
            self,
            lower: Union[int, float] = -np.inf,
            upper: Union[int, float] = np.inf
    ):
        self._lower = lower
        self._upper = upper

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(lower={self.lower}, ' \
               f'upper={self.upper})'

    @property
    def lower(self) -> Union[int, float]:
        return self._lower

    @lower.setter
    def lower(self, value: Union[int, float]) -> None:
        if isinstance(value, (int, float)):
            self._lower = value
        else:
            raise ValueError('lower limit must be int or float')

    @property
    def upper(self) -> Union[int, float]:
        return self._upper

    @upper.setter
    def upper(self, value: Union[int, float]) -> None:
        if isinstance(value, (int, float)) and value >= self.lower:
            self._upper = value
        else:
            raise ValueError('upper limit must be int or float '
                             'and more than lower')


class Quantiles(BaseLimits):
    def __init__(
            self,
            lower: Union[int, float] = 0,
            upper: Union[int, float] = 1
    ):
        if 0 <= lower <= 1 and 0 <= upper <= 1:
            super().__init__(lower, upper)
        else:
            raise ValueError('quantiles must be in range [0, 1]')


class Thresholds(BaseLimits):
    pass
