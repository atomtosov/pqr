from abc import abstractmethod

import numpy as np


class IPickingFactor:
    @abstractmethod
    def pick(self,
             data: np.ndarray,
             interval,
             looking_period: int,
             lag_period: int) -> np.ndarray:
        ...


class IFilteringFactor:
    @abstractmethod
    def filter(self, data: np.ndarray) -> np.ndarray:
        ...


class IWeightingFactor:
    @abstractmethod
    def weigh(self, data: np.ndarray) -> np.ndarray:
        ...
