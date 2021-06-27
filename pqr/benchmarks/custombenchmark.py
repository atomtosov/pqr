from typing import Optional

import pandas as pd

from .basebenchmark import BaseBenchmark
from pqr.factors.interfaces import IWeighting
from pqr.factors import EqualWeights


class CustomBenchmark(BaseBenchmark):
    def __init__(self,
                 prices: pd.DataFrame,
                 weighting_factor: Optional[IWeighting] = None,
                 name: str = ''):
        if isinstance(prices, pd.DataFrame):
            self._prices = prices.copy()
        else:
            raise ValueError('data must be pd.DataFrame')

        if isinstance(weighting_factor, IWeighting):
            self._weighting_factor = weighting_factor
        elif weighting_factor is None:
            self._weighting_factor = EqualWeights()
        else:
            raise TypeError('weighting_factor must implement IWeighting')

        if isinstance(name, str):
            self.__name = name
        else:
            raise TypeError('name must be str')

    @property
    def returns(self) -> pd.Series:
        return (
                self._prices.pct_change()
                * self._weighting_factor.weigh(self._prices)
        ).sum(axis=1)

    @property
    def _name(self) -> str:
        return self.__name
