from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd

from .interfaces import IBenchmark


class BaseBenchmark(IBenchmark, ABC):
    """
    Abstract base class for benchmarks.
    """

    returns: pd.Series

    def __repr__(self) -> str:
        """
        Dunder/Magic method for fancy printing BaseBenchmark object in console.
        """

        return f'{self.__class__.__name__}({self._name})'

    @property
    @abstractmethod
    def _name(self) -> str:
        """
        str : Name of benchmark.
        """

    def _calc_cumulative_returns(self, shift: int = 0) -> pd.Series:
        """
        Calculate cumulative returns with shift.

        First "shift" returns in series are replaced with nans to create equal
        conditions for benchmark and factor strategy: they should start invest
        in some period.

        Parameters
        ----------
        shift : int, default=0
            Non-tradable period.

        Returns
        -------
        pd.Series
            Series of cumulative (shifted) returns. Starts from 0.

        Raises
        ------
        TypeError
            Shift is not int.
        ValueError
            Shift < 0.
        """

        if not isinstance(shift, int):
            raise TypeError('shift must be int')
        elif shift < 0:
            raise ValueError('shift must be >= 0')

        returns = self.returns.copy()
        returns[:shift + 1] = 0
        return (1 + returns).cumprod() - 1

    def plot_cumulative_returns(self, shift: int = 0) -> None:
        """
        Plot cumulative (shifted) returns of benchmark.

        Parameters
        ----------
        shift : int, default=0
            Non-tradable period.
        """

        cum_returns = self._calc_cumulative_returns(shift)
        plt.plot(cum_returns.index, cum_returns.values, label=repr(self))
