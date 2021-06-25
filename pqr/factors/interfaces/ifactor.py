from typing import Optional

import numpy as np

from pqr.utils import DataPeriodicity


class IFactor:
    """
    Class-interface for factors.

    Created to pull up annotations.

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

    dynamic: bool
    bigger_better: Optional[bool]
    periodicity: DataPeriodicity

    def transform(self,
                  looking_period: int,
                  lag_period: int) -> np.ndarray:
        ...
