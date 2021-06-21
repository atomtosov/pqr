from typing import Union, Any

import numpy as np
import pandas as pd

from .basefactor import BaseFactor
from pqr.utils import lag, pct_change


class SingleFactor(BaseFactor):
    """
    Class for factors, which can be represented by 1 matrix (e.g. value - P/E).

    Parameters
    ----------
    data : np.ndarray, pd.DataFrame
        Matrix with values of factor.
    dynamic : bool, default=False
        Whether factor values should be used to make decisions in absolute form
        or in relative form (percentage changes).
    bigger_better : bool, None, default=True
        Whether more factor value, better company or less factor value better
        company. If it equals None, cannot be defined correctly (e.g. intercept
        multi-factor).
    periodicity : str, default='monthly'
        Discreteness of factor with respect to one year (e.g. 'monthly' equals
        to 12, because there are 12 trading months in 1 year).
    replace_with_nan: Any, default=None
        Value to be replaced with np.nan in data.
    name : str, optional
        Name of factor.

    Attributes
    ----------
        dynamic
        bigger_better
        periodicity
        name
    """

    _values: np.ndarray

    def __init__(self,
                 data: Union[np.ndarray, pd.DataFrame],
                 dynamic: bool = False,
                 bigger_better: bool = True,
                 periodicity: str = 'monthly',
                 replace_with_nan: Any = None,
                 name: str = None):
        """
        Initialize SingleFactor instance.
        """

        # init parent BaseFactor class
        super().__init__(dynamic, bigger_better, periodicity, name)

        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                # ensure that np.array is 2-dimensional
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
        Transform factor values into appropriate for decision-making format.

        If factor is dynamic:
            calculate percentage change t(-looking_period)/t(0)-1 and then lag
            it to lag_period+1 (additional shift is necessary, because it is
            always needed to know t(0) data, but it can be get only at t(1);
            so, it helps to avoid lookahead bias).

        If factor is static:
            just lag all values to looking_period+lag_period.

        Parameters
        ----------
        looking_period : int, default=1
            Period to lookahead in data to transform it.
        lag_period : int, default=0
            Period to lag data to create effect of delayed reaction to factor
            values.

        Returns
        -------
            2-d matrix with shape equal to shape of data with transformed
            factor values. First looking_period+lag_period lines are equal to
            np.nan, because in these moments decision-making is abandoned
            because of lack of data. For dynamic factors one more line is equal
            to np.nan (see above).
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
