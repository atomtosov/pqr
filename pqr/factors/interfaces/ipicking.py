from abc import abstractmethod

import pandas as pd

from pqr.intervals import Interval


class IPicking:
    """
    Interface for picking factors.
    """

    @abstractmethod
    def pick(self,
             data: pd.DataFrame,
             interval: Interval,
             looking_period: int,
             lag_period: int) -> pd.DataFrame:
        """
        Method, picking some stocks from stock universe by some interval on the
        basis of factor values.
        """
