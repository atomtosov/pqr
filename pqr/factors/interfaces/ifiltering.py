from abc import abstractmethod

import numpy as np


class IFiltering:
    """
    Class-interface for filtering factors.
    """

    @abstractmethod
    def filter(self, data: np.ndarray) -> np.ndarray:
        ...
