from typing import Tuple, List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pqr.factors import Factor, FilteringFactor, WeightingFactor
from pqr.portfolios import Portfolio, QuantilePortfolio
from pqr.benchmarks import Benchmark


class FactorModel:
    _portfolios: List[Portfolio]

    def __init__(self):
        self._portfolios = []

    def fit(
            self,
            prices: Union[np.ndarray, pd.DataFrame],
            factor: Factor,
            holding_period: int = 1,
            filtering_factor: FilteringFactor = None,
            weighting_factor: WeightingFactor = None,
            benchmark: Benchmark = None,
            budget: Union[int, float] = None,
            fee_rate: Union[int, float] = None,
            fee_fixed: Union[int, float] = None,
            n_quantile_portfolios: int = 3
    ) -> None:
        quantiles = np.take(
            np.linspace(0, 1, n_quantile_portfolios + 1),
            np.arange(n_quantile_portfolios * 2).
            reshape((n_quantile_portfolios, -1)) -
            np.indices((n_quantile_portfolios, 2))[0]
        )
        self._portfolios = [
            QuantilePortfolio(q, budget, fee_rate, fee_fixed).
            construct(
                prices,
                factor,
                holding_period,
                filtering_factor,
                weighting_factor,
                benchmark
            )
            for q in quantiles
        ]

    @property
    def portfolios(self) -> List[Portfolio]:
        return self._portfolios

    def compare_portfolios(self):
        stats = {}
        _ = plt.figure(figsize=(16, 9))
        for i, portfolio in enumerate(self.portfolios):
            stats[repr(portfolio)] = portfolio.stats
            portfolio.plot_cumulative_returns(
                add_benchmark=(i == len(self.portfolios) - 1)
            )
        plt.legend()
        plt.suptitle('Portfolio Cumulative Returns', fontsize=25)
        return pd.DataFrame(stats).round(2)
