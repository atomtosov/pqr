from typing import Union

import numpy as np
import pandas as pd


from .basebenchmark import BaseBenchmark
from pqr.utils import pct_change


class Benchmark(BaseBenchmark):
    def __init__(
            self,
            data: Union[np.ndarray, pd.DataFrame, pd.Series],
            name: str = None
    ):
        super().__init__(name)

        if isinstance(data, np.ndarray):
            if data.ndim != 1:
                raise ValueError('stock_index must be 1-dimensional')
            self.index = np.arange(data.shape[0])
            self._prices = data
        elif isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError('stock_index must be 1-dimensional')
            self.index = np.array(data.index.values)
            self._prices = data.values.flatten()
        elif isinstance(data, pd.Series):
            self.index = np.array(data.index.values)
            self._prices = data.values
        else:
            raise ValueError('data must be numpy.ndarray, pandas.DataFrame '
                             'or pandas.Series')

    def _calc_returns(self) -> np.ndarray:
        return pct_change(self._prices[:, np.newaxis]).flatten()
