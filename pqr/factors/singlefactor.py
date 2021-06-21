from typing import Union, Any

import numpy as np
import pandas as pd

from .basefactor import BaseFactor
from pqr.utils import lag, pct_change


class SingleFactor(BaseFactor):
    """
    Class for single factors (e.g. value - P/E)
    Inherits from BaseFactor to inherit some attributes, but overrides init

    Attributes:
        dynamic: bool - is factor dynamic or not, this information is needed
        for future transformation of factor data
        bigger_better: bool | None - is better, when factor value bigger
        (e.g. ROA) or when factor value lower (e.g. P/E); if value is None it
        means that it cannot be said exactly, what is better (used for multi-
        factors)
        periodicity: DataPeriodicity - info about periodicity or discreteness
        of factor data, used for annualization and smth more
        name: str - name of factor

    Methods:
        transform(looking_period=1, lag_period=1) - returns transformed values
        of factor data with looking_period and lag_period (NOTE: if factor is
        dynamic, real lag = lag_period + 1)
    """

    _values: np.ndarray

    def __init__(self,
                 data: Union[np.ndarray, pd.DataFrame],
                 dynamic: bool = False,
                 bigger_better: bool = True,
                 data_periodicity: str = 'monthly',
                 replace_with_nan: Any = None,
                 name: str = None):
        """
        Initialization of SingleFactor class

        :param data: matrix of factor values
        :param dynamic: is factor dynamic or not
        :param bigger_better: is better, when factor value bigger
        :param data_periodicity: periodicity or discreteness of factor data
        :param replace_with_nan: value, which interpreted as nan in data
        :param name: name of factor

        :raise ValueError if format of data values is incorrect
        """

        super().__init__(dynamic, bigger_better, data_periodicity, name)

        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                self._values = np.array(data, dtype=float)
            else:
                raise ValueError('data must be 2-dimensional')
        elif isinstance(data, pd.DataFrame):
            self._values = np.array(data.values, dtype=float)
        else:
            raise ValueError('data must be np.ndarray or pd.DataFrame')
        self._values[self._values == replace_with_nan] = np.nan

    def transform(self,
                  looking_period: int = 1,
                  lag_period: int = 0) -> np.ndarray:
        """
        Transforms factor values by looking_period and lag_period:
            - if factor is dynamic, calculate pct change to t(-looking_period)
            - if factor is static, just shift data to t(-looking_period)
            - then shift data by lag_period (if on testing you react on factor
            data with lag)

        :param looking_period: period to lookahead
        :param lag_period: period to shift data
        :return: 2-dimensional matrix of transformed factor values
        """

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
