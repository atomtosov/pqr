from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np

from pqr.factors import BaseFactor, Factor
from pqr.utils import Interval


class MultiFactor(ABC, BaseFactor):
    _factors: Tuple[Factor]

    def __init__(self,
                 factors: Sequence[Factor],
                 name: str = None):
        dynamic = any([factor.dynamic for factor in factors])
        bigger_better = all([factor.bigger_better for factor in factors])
        lower_better = all([not factor.bigger_better for factor in factors])
        super().__init__(
            dynamic,
            bigger_better or (False if lower_better else None),
            factors[0].periodicity.name,
            name
        )

        self.factors = factors

    @abstractmethod
    def choose(self,
               data: np.ndarray,
               interval: Interval,
               looking_period: int = 1,
               lag_period: int = 0) -> np.ndarray:
        ...

    @property
    def factors(self) -> Tuple[Factor]:
        return self._factors

    @factors.setter
    def factors(self, value: Sequence[Factor]) -> None:
        if np.all([isinstance(factor, Factor) for factor in value]):
            self._factors = tuple(value)
        else:
            raise ValueError('all factors must be Factor')
