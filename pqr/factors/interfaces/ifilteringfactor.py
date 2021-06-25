from abc import abstractmethod

import numpy as np


class IFilteringFactor:
    @abstractmethod
    def filter(self, data: np.ndarray) -> np.ndarray:
        ...
