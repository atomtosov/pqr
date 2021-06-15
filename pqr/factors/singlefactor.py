from typing import Union, Any

import numpy as np
import pandas as pd

from .basefactor import BaseFactor
from pqr.utils import lag, pct_change, DataPeriodicity


class SingleFactor(BaseFactor):
    def __init__(self,
                 data: Union[np.ndarray, pd.DataFrame],
                 dynamic: bool = False,
                 bigger_better: bool = True,
                 data_periodicity: str = 'monthly',
                 replace_with_nan: Any = None,
                 name: str = None):
        super().__init__(dynamic, bigger_better, data_periodicity, name)

        if isinstance(data, np.ndarray):
            assert data.ndim == 2, 'data must be 2-dimensional'
            self._values = np.array(data, dtype=float)
        elif isinstance(data, pd.DataFrame):
            self._values = np.array(data.values, dtype=float)
        else:
            raise ValueError('data must be np.ndarray or pd.DataFrame')
        self._values[self._values == replace_with_nan] = np.nan

    def transform(self,
                  looking_period: int = 1,
                  lag_period: int = 0) -> np.ndarray:
        if not isinstance(looking_period, int) or looking_period < 1:
            raise ValueError('looking_period must be int >= 1')
        if not isinstance(lag_period, int) or lag_period < 0:
            raise ValueError('lag_period must be int >= 0')

        if self.dynamic:
            return lag(
                pct_change(self._values, looking_period),
                lag_period + 1
            )
        else:
            return lag(
                self._values,
                looking_period + lag_period
            )
