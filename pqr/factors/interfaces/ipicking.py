from abc import abstractmethod

import numpy as np

from pqr.intervals import Interval


class IPicking:
    """
    Class-interface for picking factors.
    """

    @abstractmethod
    def pick(self,
             data: np.ndarray,
             interval: Interval,
             looking_period: int,
             lag_period: int) -> np.ndarray:
        ...
