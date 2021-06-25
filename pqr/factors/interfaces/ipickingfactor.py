from abc import abstractmethod

import numpy as np

from .ifactor import IFactor
from pqr.utils import Interval


class IPickingFactor(IFactor):
    """
    Class-interface for picking factors.

    Attributes
    ----------
    dynamic : bool
        Whether factor values should be used to make decisions in absolute form
        or in relative form (percentage changes).
    bigger_better : bool, None
        Whether more factor value, better company or less factor value better
        company. If it equals None, cannot be defined correctly (e.g. intercept
        multi-factor).
    periodicity : str
        Discreteness of factor with respect to one year (e.g. 'monthly' equals
        to 12, because there are 12 trading months in 1 year).
    """

    @abstractmethod
    def pick(self,
             data: np.ndarray,
             interval: Interval,
             looking_period: int,
             lag_period: int) -> np.ndarray:
        ...
