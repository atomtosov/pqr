from abc import abstractmethod

import numpy as np


class IWeightingFactor:
    @abstractmethod
    def weigh(self, data: np.ndarray) -> np.ndarray:
        ...
