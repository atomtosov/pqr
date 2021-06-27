from typing import Optional

import pandas as pd

from ..basefactor import BaseFactor


class SingleFactor(BaseFactor):
    """
    Class for factors, which can be represented by 1 matrix (e.g. value - P/E).

    Parameters
    ----------
    data : pd.DataFrame
        Matrix with values of factor.
    dynamic : bool, default=False
        Whether factor values should be used to make decisions in absolute form
        or in relative form (percentage changes).
    bigger_better : bool, None, default=True
        Whether more factor value, better company or less factor value better
        company. If it equals None, cannot be defined correctly (e.g. intercept
        multi-factor).
    name : str, optional
        Name of factor.

    Attributes
    ----------
        dynamic
        bigger_better
        name
    """

    def __init__(self,
                 data: pd.DataFrame,
                 dynamic: bool = False,
                 bigger_better: bool = True,
                 name: str = ''):
        """
        Initialize SingleFactor instance.
        """

        if isinstance(data, pd.DataFrame):
            self._data = data.copy()
        else:
            raise TypeError('data must be pd.DataFrame')

        if isinstance(dynamic, bool):
            self._dynamic = dynamic
        else:
            raise TypeError('dynamic must be bool')

        if isinstance(bigger_better, bool):
            self._bigger_better = bigger_better
        else:
            raise TypeError('bigger_better must be int')

        if isinstance(name, str):
            self._name = name
        else:
            raise TypeError('name must be str')

    def transform(self,
                  looking_period: int = 1,
                  lag_period: int = 0) -> pd.DataFrame:
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
            return self._data.pct_change(looking_period).shift(lag_period + 1)
        else:
            return self._data.shift(looking_period + lag_period)

    @property
    def dynamic(self) -> bool:
        return self._dynamic

    @property
    def bigger_better(self) -> Optional[bool]:
        return self._bigger_better

    @property
    def name(self) -> str:
        return self._name
