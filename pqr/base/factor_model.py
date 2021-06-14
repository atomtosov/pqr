from abc import ABC, abstractmethod

from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .portfolio import BasePortfolio
from .limits import Quantiles
from pqr.utils import make_intervals


class BaseFactorModel(ABC):
    _portfolios: List[BasePortfolio]

    def __init__(self):
        self._portfolios = []

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        ...

    @staticmethod
    def _get_quantiles(n: int = 1) -> List[Quantiles]:
        return [
            Quantiles(*interval)
            for interval in make_intervals(
                np.linspace(0, 1, n + 1)
            )
        ]

    @staticmethod
    def _get_thresholds(n: int = 1):
        ...

    @property
    def portfolios(self) -> List[BasePortfolio]:
        return self._portfolios

    def compare_portfolios(self):
        stats = {}
        plt.figure(figsize=(16, 9))
        for i, portfolio in enumerate(self.portfolios):
            stats[repr(portfolio)] = portfolio.stats
            portfolio.plot_cumulative_returns(
                add_benchmark=(i == len(self.portfolios) - 1)
            )
        plt.legend()
        plt.suptitle('Portfolio Cumulative Returns', fontsize=25)
        return pd.DataFrame(stats).round(2)
