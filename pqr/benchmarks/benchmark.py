import pandas as pd

from .basebenchmark import BaseBenchmark


class Benchmark(BaseBenchmark):
    def __init__(self,
                 values: pd.Series,
                 name: str = ''):
        if isinstance(values, pd.Series):
            self._values = values.copy()
        else:
            raise TypeError('prices must be pandas.Series')

        self.__name = name

    @property
    def returns(self) -> pd.Series:
        return self._values.pct_change()

    @property
    def _name(self) -> str:
        return self.__name
