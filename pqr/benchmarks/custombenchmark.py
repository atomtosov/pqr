from typing import Optional

import pandas as pd

from .basebenchmark import BaseBenchmark
from pqr.factors.interfaces import IWeighting
from pqr.factors import EqualWeights


class CustomBenchmark(BaseBenchmark):
    """
    Class for making custom indices-benchmarks from prices.
    """

    def __init__(self,
                 prices: pd.DataFrame,
                 weighting_factor: Optional[IWeighting] = None,
                 name: str = ''):
        """
        Initialize CustomBenchmark instance.

        Parameters
        ----------
        prices : pd.DataFrame
            Prices of stocks to be picked. All of available stocks are in the
            portfolio in each period.
        weighting_factor : IWeighting, optional
            Factor to weigh stocks in portfolio (e.g. market capitalization).
            If not passed, equal weights are used.
        name : str
            Name of benchmark.

        Raises
        ------
        TypeError
            Given prices is not pd.DataFrame, or weighting_factor doesn't
            implement weighting interface, or name is not str.
        """

        if not isinstance(prices, pd.DataFrame):
            raise TypeError('prices must be pd.DataFrame')

        if weighting_factor is None:
            weighting_factor = EqualWeights()
        elif not isinstance(weighting_factor, IWeighting):
            raise TypeError('weighting_factor must implement IWeighting')

        self._returns = (
                weighting_factor.weigh(prices)
                * prices.pct_change().shift(-1)
        ).shift().sum(axis=1)

        if isinstance(name, str):
            self.__name = name
        else:
            raise TypeError('name must be str')

    @property
    def returns(self) -> pd.Series:
        return self._returns

    @property
    def _name(self) -> str:
        return self.__name
