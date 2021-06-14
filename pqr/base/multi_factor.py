from abc import ABC
from typing import Tuple, Iterable, Union

import numpy as np

from .factor import BaseFactor, \
    ChoosingFactorInterface, \
    FilteringFactorInterface, \
    WeightingFactorInterface


class BaseMultiFactor(ABC):
    _factors: Tuple[Union[BaseFactor,
                          ChoosingFactorInterface,
                          FilteringFactorInterface,
                          WeightingFactorInterface]]

    def __init__(self, factors: Iterable[BaseFactor]):
        self.factors = factors

    @property
    def factors(self) -> Tuple[Union[BaseFactor,
                               ChoosingFactorInterface,
                               FilteringFactorInterface,
                               WeightingFactorInterface]]:
        return self._factors

    @factors.setter
    def factors(self, value: Iterable[Union[BaseFactor,
                                      ChoosingFactorInterface,
                                      FilteringFactorInterface,
                                      WeightingFactorInterface]]):
        if np.all(isinstance(factor, (BaseFactor,
                                      ChoosingFactorInterface,
                                      FilteringFactorInterface,
                                      WeightingFactorInterface))
                  for factor in value):
            self._factors = tuple(value)
        else:
            raise ValueError('all factors must be BaseFactor')

    @property
    def shift(self):
        return int(np.max([factor.looking_period for factor in self.factors]) + \
               np.max([factor.lag for factor in self.factors]) + \
               np.any([not factor.static for factor in self.factors]))

    @property
    def periodicity(self):
        return self.factors[0].periodicity
