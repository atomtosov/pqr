from abc import abstractmethod

import numpy as np


class IWeighting:
    """
    Class-interface for weighting factors.
    """

    @abstractmethod
    def weigh(self, data: np.ndarray) -> np.ndarray:
        ...
