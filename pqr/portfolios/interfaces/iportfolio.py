from abc import abstractmethod
from typing import Optional

import pandas as pd

from pqr.factors.interfaces import IPicking, IFiltering, IWeighting
from pqr.benchmarks.interfaces import IBenchmark


class IPortfolio:
    """
    Class-interface for portfolio.
    """

    positions: pd.DataFrame
    returns: pd.Series

    stats: pd.DataFrame

    @abstractmethod
    def invest(self,
               prices: pd.DataFrame,
               picking_factor: IPicking,
               looking_period: int = 1,
               lag_period: int = 0,
               holding_period: int = 1,
               filtering_factor: Optional[IFiltering] = None,
               weighting_factor: Optional[IWeighting] = None,
               benchmark: Optional[IBenchmark] = None) -> None:
        """
        Method for investing by factor.
        """

    @abstractmethod
    def plot_cumulative_returns(self, add_benchmark=False) -> None:
        ...
