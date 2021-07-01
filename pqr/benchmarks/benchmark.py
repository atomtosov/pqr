import pandas as pd

from .basebenchmark import BaseBenchmark


class Benchmark(BaseBenchmark):
    """
    Class for existing benchmarks (e.g. S&P500).
    """

    def __init__(self,
                 values: pd.Series,
                 name: str = ''):
        """
        Initialize Benchmark instance.

        Parameters
        ----------
        values : pd.Series
            Values of some benchmark-index. Percentage changes of that values
            are used as returns of benchmark.
        name : str, optional
            Name of benchmark.
        """

        if isinstance(values, pd.Series):
            self._returns = values.pct_change()
        else:
            raise TypeError('values must be pd.Series')

        self.__name = name

    @property
    def returns(self) -> pd.Series:
        return self._returns

    @property
    def _name(self) -> str:
        return self.__name
