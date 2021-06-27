from abc import abstractmethod

import pandas as pd

from pqr.intervals import Interval


class IPicking:
    """
    Class-interface for picking factors.
    """

    @abstractmethod
    def pick(self,
             data: pd.DataFrame,
             interval: Interval,
             looking_period: int,
             lag_period: int) -> pd.DataFrame:
        ...
