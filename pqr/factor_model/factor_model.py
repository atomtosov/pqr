from typing import Union, List, Iterable, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pqr.factors import Factor, FilteringFactor, WeightingFactor
from pqr.multi_factors import MultiFactor
from pqr.portfolios import BasePortfolio, QuantilePortfolio, WMLPortfolio
from pqr.benchmarks import BaseBenchmark
from pqr.utils import HasNameMixin, make_intervals, Quantiles


class FactorModel(HasNameMixin):
    _looking_period: int
    _lag_period: int
    _holding_period: int

    _portfolios: List[BasePortfolio]

    def __init__(self,
                 looking_period: int = 1,
                 lag_period: int = 0,
                 holding_period: int = 1):
        super().__init__(f'{looking_period}, {lag_period}, {holding_period}')

        self.looking_period = looking_period
        self.lag_period = lag_period
        self.holding_period = holding_period

        self._portfolios = []

    def fit(self,
            prices: Union[np.ndarray, pd.DataFrame],
            factor: Union[Factor, MultiFactor],
            filtering_factor: FilteringFactor = None,
            weighting_factor: WeightingFactor = None,
            benchmark: BaseBenchmark = None,
            budget: Union[int, float] = None,
            fee_rate: Union[int, float] = None,
            fee_fixed: Union[int, float] = None,
            n_quantiles: int = 3,
            add_wml: bool = False):
        if not isinstance(n_quantiles, int) or n_quantiles <= 0:
            raise ValueError('n_quantiles must be int > 0')
        quantiles = [
            Quantiles(*pair) for pair
            in make_intervals(np.linspace(0, 1, n_quantiles+1))
        ]
        self._portfolios = [
            QuantilePortfolio(q, budget, fee_rate, fee_fixed)
            for q in quantiles
        ]
        for portfolio in self._portfolios:
            portfolio.construct(
                prices,
                factor,
                self.looking_period,
                self.lag_period,
                self.holding_period,
                filtering_factor,
                weighting_factor,
                benchmark
            )
        if add_wml:
            wml = WMLPortfolio()
            if factor.bigger_better or factor.bigger_better is None:
                wml.construct(winners=self._portfolios[-1],
                              losers=self._portfolios[0],
                              benchmark=benchmark)
            else:
                wml.construct(winners=self._portfolios[0],
                              losers=self._portfolios[-1],
                              benchmark=benchmark)
            self._portfolios.append(wml)

    def compare_portfolios(self, plot: bool = True):
        stats = {}
        plt.figure(figsize=(16, 9))
        for i, portfolio in enumerate(self.portfolios):
            stats[repr(portfolio)] = portfolio.stats
            if plot:
                portfolio.plot_cumulative_returns(
                    add_benchmark=(i == len(self.portfolios) - 1)
                )
        if plot:
            plt.legend()
            plt.suptitle('Portfolio Cumulative Returns', fontsize=25)
        return pd.DataFrame(stats).round(2)

    def grid_search(self,
                    looking_periods: Iterable[int],
                    lag_periods: Iterable[int],
                    holding_periods: Iterable[int],
                    prices: Union[np.ndarray, pd.DataFrame],
                    factor: Union[Factor, MultiFactor],
                    filtering_factor: FilteringFactor = None,
                    weighting_factor: WeightingFactor = None,
                    benchmark: BaseBenchmark = None,
                    budget: Union[int, float] = None,
                    fee_rate: Union[int, float] = None,
                    fee_fixed: Union[int, float] = None,
                    n_quantiles: int = 3,
                    add_wml: bool = False) -> Dict[Tuple[int, int, int],
                                                   pd.DataFrame]:
        results = {}
        for looking_period in looking_periods:
            for lag_period in lag_periods:
                for holding_period in holding_periods:
                    self.looking_period = looking_period
                    self.lag_period = lag_period
                    self.holding_period = holding_period
                    self.fit(
                        prices,
                        factor,
                        filtering_factor,
                        weighting_factor,
                        benchmark,
                        budget,
                        fee_rate,
                        fee_fixed,
                        n_quantiles,
                        add_wml
                    )
                    results[(looking_period, lag_period, holding_period)] = \
                        self.compare_portfolios(plot=False)
        return results

    @property
    def portfolios(self) -> List[BasePortfolio]:
        return self._portfolios

    @property
    def looking_period(self) -> int:
        return self._looking_period

    @looking_period.setter
    def looking_period(self, value: int) -> None:
        if isinstance(value, int) and value >= 1:
            self._looking_period = value
        else:
            raise ValueError('looking_period must be int >= 1')

    @property
    def lag_period(self) -> int:
        return self._lag_period

    @lag_period.setter
    def lag_period(self, value: int):
        if isinstance(value, int) and value >= 0:
            self._lag_period = value
        else:
            raise ValueError('lag_period must be int >= 0')

    @property
    def holding_period(self) -> int:
        return self._holding_period

    @holding_period.setter
    def holding_period(self, value: int) -> None:
        if isinstance(value, int) and value >= 1:
            self._holding_period = value
        else:
            raise ValueError('holding_period must be int >= 1')
