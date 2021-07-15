from typing import List, Tuple, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pqr.benchmarks
import pqr.factors
import pqr.metrics
import pqr.portfolios
import pqr.thresholds
import pqr.visualization


__all__ = [
    'FactorModel',
    'grid_search',
]


class FactorModel:
    """
    Class for building factor models: covering all stock universe with
    portfolios.

    Parameters
    ----------
    looking_period : int, default=1
        Looking back period.
    lag_period : int, default=0
        Delaying period to react on factor.
    holding_period : int, default=1
        Number of periods to hold each position in portfolios.
    """

    portfolios: List[pqr.portfolios.AbstractPortfolio]

    def __init__(self,
                 looking_period: int = 1,
                 lag_period: int = 0,
                 holding_period: int = 1):
        self.looking_period = looking_period
        self.lag_period = lag_period
        self.holding_period = holding_period

    def __repr__(self) -> str:
        return (f'{type(self).__name__}({self.looking_period}-'
                f'{self.lag_period}-{self.holding_period})')

    def fit(self,
            prices: pd.DataFrame,
            picking_factor: pqr.factors.Factor,
            balance: Union[int, float] = None,
            fee_rate: Union[int, float] = 0,
            n_quantiles: int = 3,
            add_wml: bool = False,
            **kwargs) -> 'FactorModel':
        """
        Creates factor portfolios, covering all stock universe.

        Parameters
        ----------
        prices : pd.DataFrame
            Dataframe, representing stock universe by prices.
        picking_factor : pqr.Factor
            Factor, used to pick stocks from (filtered) stock universe.
        balance : int, default=None
            Initial balance of portfolio. If not given, used relative
            portfolio-building.
        fee_rate : int, float, default=0
            Commission rate for every deal. For relative portfolio is not used.
        n_quantiles : int, default=3
            Number of portfolios to build for covering stock universe.
        add_wml : bool, default=False
            Whether to add wml-portfolio or not.
        **kwargs : dict
            Keyword arguments for portfolio investing. See
            pqr.portfolios.Portfolio.invest() parameters.

        Returns
        -------
        FactorModel
            The same object, but with filled portfolios.
        """

        quantiles = _make_quantiles(n_quantiles)
        self.portfolios = [
            pqr.portfolios.Portfolio(balance, fee_rate).invest(
                prices,
                picking_factor, quantile,
                self.looking_period, self.lag_period, self.holding_period,
                **kwargs
            )
            for quantile in quantiles
        ]

        if add_wml:
            wml = pqr.portfolios.WmlPortfolio()
            if picking_factor.bigger_better:
                wml.invest(self.portfolios[-1], self.portfolios[0])
            else:
                wml.invest(self.portfolios[0], self.portfolios[1])
            self.portfolios.append(wml)

        return self

    def compare_portfolios(self,
                           benchmark: pqr.benchmarks.AbstractBenchmark,
                           plot: bool = True) -> pd.DataFrame:
        stats = []
        for portfolio in self.portfolios:
            stats.append(pqr.metrics.summary(portfolio, benchmark))

        if plot:
            pqr.visualization.plot_cumulative_returns(*self.portfolios,
                                                      benchmark=benchmark)
        return pd.DataFrame(stats).T.round(2)


def grid_search(looking_periods: List[int],
                lag_periods: List[int],
                holding_periods: List[int],
                benchmark: pqr.benchmarks.AbstractBenchmark,
                **kwargs) -> Dict[Tuple[int, int, int],
                                               pd.DataFrame]:
    """
    Fits factor models by grid of looking, lag and holding periods and save
    their statistics. Can be used to find the best parameters or just as fast
    alias to build a lot of models.

    Parameters
    ----------
    looking_periods : list of int
        Looking periods to be tested.
    lag_periods : list of int
        Lag periods to be tested.
    holding_periods : list of int
        Holding periods to be tested.
    benchmark : AbstractBenchmark
        Benchmark to compare with portfolios and calculate metrics and
        statistics.
    **kwargs : dict
        Keyword arguments for fitting factor models. See
        pqr.factor_model.FactorModel.fit() parameters.

    Returns
    -------
    Dict[Tuple[int, int, int], pd.DataFrame]
        Dict, where key is tuple of (looking_period, lag_period,
        holding_period) and value is dataframe with summary statistics for the
        factor model with such parameters.
    """

    results = {}
    for looking in looking_periods:
        for lag in lag_periods:
            for holding in holding_periods:
                fm = FactorModel(looking, lag, holding)
                fm.fit(**kwargs)
                results[(looking, lag, holding)] = fm.compare_portfolios(
                    benchmark, plot=False)
    return results


def _make_quantiles(n: int) -> List[pqr.thresholds.Quantiles]:
    """
    Makes quantiles, covering all range from 0 to 1.

    Parameters
    ----------
    n : int
        How many quantiles to construct.

    Returns
    -------
    List of Quantiles
        List of "n" quantiles, covering all range from 0 to 1.
    """

    quantile_pairs = np.take(np.linspace(0, 1, n + 1),
                             np.arange(n * 2).reshape((n, -1)) -
                             np.indices((n, 2))[0])
    return [pqr.thresholds.Quantiles(*pair)
            for pair in quantile_pairs]
