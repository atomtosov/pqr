from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BaseBenchmark:
    """
    Abstract base class for benchmarks.

    Attributes
    ----------
    returns
    cumulative_returns
    """

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._name})'

    @property
    @abstractmethod
    def returns(self) -> pd.Series:
        ...

    @property
    @abstractmethod
    def _name(self) -> str:
        ...

    def calc_cumulative_returns(self, shift: int = 0) -> pd.Series:
        if not isinstance(shift, int) or shift < 0:
            raise ValueError('shift must be int > 0')

        returns = self.returns
        returns[:shift + 1] = np.nan
        return (returns + 1).cumprod() - 1

    @property
    def cumulative_returns(self):
        return self.calc_cumulative_returns()

    def plot_cumulative_returns(self, shift: int = 0):
        if not isinstance(shift, int):
            raise TypeError('shift must be int')
        elif shift < 0:
            raise ValueError('shift must be >= 0')

        cum_returns = self.calc_cumulative_returns(shift)
        plt.plot(cum_returns.index, cum_returns.values, label=repr(self))
