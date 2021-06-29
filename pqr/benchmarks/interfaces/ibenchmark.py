from abc import abstractmethod

import pandas as pd


class IBenchmark:
    @property
    @abstractmethod
    def returns(self) -> pd.Series:
        ...

    @abstractmethod
    def plot_cumulative_returns(self) -> None:
        ...
