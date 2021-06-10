from typing import Union

import numpy as np
import pandas as pd

from pqr.utils import pct_change


class Benchmark:
    _index: np.ndarray
    _returns: np.ndarray

    def __init__(
            self,
            stock_index: Union[np.ndarray, pd.DataFrame, pd.Series]
    ):
        if isinstance(stock_index, np.ndarray):
            if stock_index.ndim != 1:
                raise ValueError('stock_index must be 1-dimensional')
            self._index = np.arange(stock_index.shape[0])
            self._returns = pct_change(stock_index[:, np.newaxis]).flatten()
        elif isinstance(stock_index, pd.DataFrame):
            if stock_index.shape[1] != 1:
                raise ValueError('stock_index must be 1-dimensional')
            self._index = np.array(stock_index.index.values)
            self._returns = pct_change(stock_index.values).flatten()
        elif isinstance(stock_index, pd.Series):
            self._index = np.array(stock_index.index.values)
            self._returns = pct_change(stock_index.values[:, np.newaxis])\
                .flatten()

    @property
    def index(self) -> np.ndarray:
        return self._index

    @property
    def returns(self) -> np.ndarray:
        return self._returns

    @property
    def total_returns(self) -> np.ndarray:
        return np.nansum(self.returns, axis=1)

    @property
    def cumulative_returns(self):
        return np.nancumsum(self.total_returns)
