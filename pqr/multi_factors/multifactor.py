from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np

from pqr.factors import BaseFactor, Factor
from pqr.utils import Interval


class MultiFactor(ABC, BaseFactor):
    """
    Abstract base class for multi-factors, which consist of some factors.

    Parameters
    ----------
    factors : sequence of Factor
        Sequence of factors.
    name : str, optional
        Name of factor.

    Attributes
    ----------
        dynamic
        bigger_better
        periodicity
        name
        factors
    """

    _factors: Tuple[Factor]

    def __init__(self,
                 factors: Sequence[Factor],
                 name: str = None):
        # dynamic if only one factor is dynamic
        dynamic = any([factor.dynamic for factor in factors])
        # bigger_better if all factors are bigger_better
        bigger_better = all([factor.bigger_better for factor in factors])
        # lower_better if all factors are lower_better
        lower_better = all([not factor.bigger_better for factor in factors])
        # init parent ABC
        ABC.__init__(self)
        # init parent BaseFactor
        BaseFactor.__init__(
            self,
            dynamic,
            # if not bigger better and not lower_better, than None
            bigger_better or (False if lower_better else None),
            factors[0].periodicity.name,
            name
        )

        self.factors = factors

    @abstractmethod
    def pick(self,
             data: np.ndarray,
             interval: Interval,
             looking_period: int = 1,
             lag_period: int = 0) -> np.ndarray:
        """
        Pick stocks from data, using some interval.

        Must provide the same interface as Factor.pick().
        """

    @property
    def factors(self) -> Tuple[Factor]:
        return self._factors

    @factors.setter
    def factors(self, value: Sequence[Factor]) -> None:
        if np.all([isinstance(factor, Factor) for factor in value]):
            self._factors = tuple(value)
        else:
            raise ValueError('all factors must be Factor')
