import pandas as pd

from ..basefactor import BaseFactor


class SingleFactor(BaseFactor):
    """
    Class for single-factors, so factors with only one matrix of values.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 dynamic: bool = False,
                 bigger_better: bool = True,
                 name: str = ''):
        """
        Initialize SingleFactor instance.

        Parameters
        ----------
        data : pd.DataFrame
            Matrix with values of factor. Must be numeric (contains only
            int/float numbers and nans).
        dynamic : bool, default=False
            Is to interpret factor values statically (values itself) or
            dynamically (changes of values).
        bigger_better : bool, default=True
            Is bigger factor values are responsible for more attractive company
            or vice versa.
        name : str, optional
            Name of factor.

        Raises
        ------
        TypeError
            Given data is not pd.DataFrame, dynamic or bigger_better is not
            bool, or name is not str.
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
            raise TypeError('bigger_better must be bool or None')

        if isinstance(name, str):
            self.__name = name
        else:
            raise TypeError('name must be str')

    def transform(self,
                  looking_period: int = 1,
                  lag_period: int = 0) -> pd.DataFrame:
        """
        Transform factor values into appropriate for decision-making format.

        If factor is dynamic:
            calculate percentage changes with looking back for "looking_period"
            periods, then values are lagged for 1 period (because in period
            t(0) we can know percentage change from period t(-looking_period)
            only at the end of t(0), so it is needed to avoid looking-forward
            bias); then values are lagged for lag_period.

        If factor is static:
            all values are lagged for the sum of "looking_period" and
            "lag_period".

        Parameters
        ----------
        looking_period : int, default=1
            Period to look back for making decisions.
        lag_period : int, default=0
            Period of delaying entry into positions.

        Returns
        -------
        pd.DataFrame
            DataFrame with the same shape as given data, but with first rows,
            filled with nans. The amount of "blank" rows depends on the sum of
            "looking_period", "lag_period" and indicator that factor is
            dynamic.

        Raises
        ------
        TypeError
            looking_period or lag_period is not int.
        ValueError
            looking_period < 1 or lag_period < 0 or the sum of them and
            indicator of factor dynamics exceeds the number of observations
            in factor data.
        """

        if not isinstance(looking_period, int):
            raise TypeError('looking_period must be int')
        elif looking_period < 1:
            raise ValueError('looking_period must be >= 1')

        if not isinstance(lag_period, int):
            raise TypeError('lag_period must be int')
        elif lag_period < 0:
            raise ValueError('lag_period must be >= 0')

        if looking_period + lag_period + self.dynamic >= self._data:
            raise ValueError('the sum of looking_period, lag_period and '
                             '1 if factor is dynamic must be less than periods'
                             'of factor values')

        if self.dynamic:
            return self._data.pct_change(looking_period).shift(1 + lag_period)
        else:
            return self._data.shift(looking_period + lag_period)

    @property
    def dynamic(self) -> bool:
        return self._dynamic

    @property
    def bigger_better(self) -> bool:
        return self._bigger_better

    @property
    def _name(self) -> str:
        return self.__name
