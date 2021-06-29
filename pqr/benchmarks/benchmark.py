import pandas as pd

from .basebenchmark import BaseBenchmark


class Benchmark(BaseBenchmark):
    """
    Class for existing benchmarks (e.g. S&P500).

    Parameters
    ----------
    values : pd.Series
        Values of some benchmark-index.
    name : str, optional
        Name of index.

    Attributes
    ----------
    returns
    """

    def __init__(self,
                 values: pd.Series,
                 name: str = ''):
        """
        Initialize Benchmark instance.
        """

        if isinstance(values, pd.Series):
            self._values = values.copy()
        else:
            raise TypeError('prices must be pd.Series')

        self.__name = name

    @property
    def returns(self) -> pd.Series:
        return self._values.pct_change()

    @property
    def _name(self) -> str:
        return self.__name
