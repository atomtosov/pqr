from typing import Optional, Iterable, Dict, Tuple, List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pqr.factors.interfaces import IPicking, IFiltering, IWeighting
from pqr.portfolios.interfaces import IPortfolio, IWMLPortfolio
from pqr.benchmarks.interfaces import IBenchmark

from pqr.portfolios import Portfolio, WMLPortfolio
from pqr.benchmarks import CustomBenchmark
from pqr.intervals import Quantiles


class FactorModel:
    looking_period: int
    lag_period: int
    holding_period: int

    portfolios: Tuple[Union[IPortfolio, IWMLPortfolio], ...]

    def __init__(self,
                 looking_period: int = 1,
                 lag_period: int = 0,
                 holding_period: int = 1):
        if not isinstance(looking_period, int):
            raise TypeError('looking_period must be int')
        elif looking_period < 1:
            raise ValueError('looking_period must be >= 1')
        else:
            self._looking_period = looking_period

        if not isinstance(lag_period, int):
            raise TypeError('lag_period must be int')
        elif lag_period < 0:
            raise ValueError('lag_period must be >= 0')
        else:
            self._lag_period = lag_period

        if not isinstance(holding_period, int):
            raise TypeError('holding_period must be int')
        elif holding_period < 1:
            raise ValueError('holding_period must be >= 1')
        else:
            self._holding_period = holding_period

        self._portfolios = tuple()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(' \
               f'{self.looking_period}, ' \
               f'{self.lag_period}, ' \
               f'{self.holding_period}' \
               f')'

    def fit(self,
            prices: pd.DataFrame,
            picking_factor: IPicking,
            filtering_factor: Optional[IFiltering] = None,
            weighting_factor: Optional[IWeighting] = None,
            benchmark: Optional[IBenchmark] = None,
            n_quantiles: int = 3,
            add_wml: bool = False):
        if not isinstance(n_quantiles, int):
            raise TypeError('n_quantiles must be int')
        elif n_quantiles <= 0:
            raise ValueError('n_quantiles must be > 0')

        if benchmark is None:
            benchmark = CustomBenchmark(prices)

        quantiles = self._get_quantiles(n_quantiles)
        portfolios = [Portfolio(q) for q in quantiles]
        for portfolio in portfolios:
            portfolio.invest(
                prices,
                picking_factor,
                self.looking_period,
                self.lag_period,
                self.holding_period,
                filtering_factor,
                weighting_factor,
                benchmark
            )

        if add_wml:
            wml = WMLPortfolio()
            if picking_factor.bigger_better \
                    or picking_factor.bigger_better is None:
                wml.invest(winners=portfolios[-1],
                           losers=portfolios[0])
            else:
                wml.invest(winners=portfolios[0],
                           losers=portfolios[-1])
            portfolios.append(wml)

        self._portfolios = tuple(portfolios)

    def compare_portfolios(self, plot: bool = True) -> pd.DataFrame:
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

    @classmethod
    def grid_search(cls,
                    looking_periods: Iterable[int],
                    lag_periods: Iterable[int],
                    holding_periods: Iterable[int],
                    prices: pd.DataFrame,
                    picking_factor: IPicking,
                    filtering_factor: Optional[IFiltering] = None,
                    weighting_factor: Optional[IWeighting] = None,
                    benchmark: Optional[IBenchmark] = None,
                    n_quantiles: int = 3,
                    add_wml: bool = False) -> Dict[Tuple[int, int, int],
                                                   pd.DataFrame]:
        results = {}
        for looking_period in looking_periods:
            for lag_period in lag_periods:
                for holding_period in holding_periods:
                    fm = cls(looking_period, lag_period, holding_period)
                    fm.fit(
                        prices,
                        picking_factor,
                        filtering_factor,
                        weighting_factor,
                        benchmark,
                        n_quantiles,
                        add_wml
                    )
                    results[(looking_period, lag_period, holding_period)] = \
                        fm.compare_portfolios(plot=False)
        return results

    @staticmethod
    def _make_intervals(array: np.ndarray):
        assert array.ndim == 1, 'array must be 1-dimensional'
        n = np.size(array) - 1
        return np.take(
            array,
            np.arange(n * 2).reshape((n, -1)) - np.indices((n, 2))[0]
        )

    def _get_quantiles(self, n) -> List[Quantiles]:
        return [Quantiles(*pair)
                for pair in self._make_intervals(np.linspace(0, 1, n+1))]

    @property
    def looking_period(self) -> int:
        return self._looking_period

    @property
    def lag_period(self) -> int:
        return self._lag_period

    @property
    def holding_period(self) -> int:
        return self._holding_period

    @property
    def portfolios(self) -> Tuple[Union[IPortfolio, IWMLPortfolio], ...]:
        return self._portfolios
