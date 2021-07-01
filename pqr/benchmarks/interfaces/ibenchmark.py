from abc import abstractmethod

import pandas as pd


class IBenchmark:
    """
    Interface for benchmarks.
    """

    @property
    @abstractmethod
    def returns(self) -> pd.Series:
        """
        pd.Series : Series of period-to-period returns of benchmark.
        """

    @abstractmethod
    def plot_cumulative_returns(self, shift: int) -> None:
        """
        Method for plotting cumulative returns of benchmark with shift (
        non-tradable period of main strategy).
        """
