from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from pqr.core import Portfolio, PortfolioBuilder
from pqr.utils import align_many

__all__ = [
    "ProphetTest",
]


@dataclass
class ProphetTest:
    universe: pd.DataFrame
    prices: pd.DataFrame
    portfolio_builder: PortfolioBuilder
    target_metric: Callable[[Portfolio], pd.Series]

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        best_portfolio = self.portfolio_builder(
            *BestPickingStrategy(self.universe, self.prices)(portfolio)
        )
        worst_portfolio = self.portfolio_builder(
            *WorstPickingStrategy(self.universe, self.prices)(portfolio)
        )

        best_metric = self.target_metric(best_portfolio)
        worst_metric = self.target_metric(worst_portfolio)
        portfolio_metric = self.target_metric(portfolio)

        return (portfolio_metric - worst_metric) / (best_metric - worst_metric)


@dataclass
class BestPickingStrategy:
    universe: pd.DataFrame
    prices: pd.DataFrame

    def __call__(self, portfolio: Portfolio) -> tuple[pd.DataFrame, pd.DataFrame]:
        longs, shorts = portfolio.get_longs(), portfolio.get_shorts()
        longs, shorts, universe, prices = align_many(longs, shorts, self.universe, self.prices)

        long_limits = longs.sum(axis=1)
        short_limits = shorts.sum(axis=1)

        universe_returns = prices.pct_change().shift(-1)
        universe_returns[~universe] = -1

        longs = pd.DataFrame(
            np.zeros_like(longs, dtype=bool),
            index=longs.index,
            columns=longs.columns
        )
        shorts = pd.DataFrame(
            np.zeros_like(shorts, dtype=bool),
            index=shorts.index,
            columns=shorts.columns
        )
        for date in universe.index:
            best = universe_returns.loc[date].nlargest(long_limits.loc[date]).index
            worst = universe_returns.loc[date].nsmallest(short_limits.loc[date]).index

            longs.loc[date, best] = True
            shorts.loc[date, worst] = True

        return longs, shorts


@dataclass
class WorstPickingStrategy:
    universe: pd.DataFrame
    prices: pd.DataFrame

    def __call__(self, portfolio: Portfolio) -> tuple[pd.DataFrame, pd.DataFrame]:
        longs, shorts = portfolio.get_longs(), portfolio.get_shorts()
        longs, shorts, universe, prices = align_many(longs, shorts, self.universe, self.prices)

        long_limits = longs.sum(axis=1)
        short_limits = shorts.sum(axis=1)

        universe_returns = prices.pct_change().shift(-1)

        longs = pd.DataFrame(
            np.zeros_like(longs, dtype=bool),
            index=longs.index,
            columns=longs.columns
        )
        shorts = pd.DataFrame(
            np.zeros_like(shorts, dtype=bool),
            index=shorts.index,
            columns=shorts.columns
        )
        for date in universe.index:
            best = universe_returns.loc[date].nsmallest(long_limits.loc[date]).index
            worst = universe_returns.loc[date].nlargest(short_limits.loc[date]).index

            longs.loc[date, best] = True
            shorts.loc[date, worst] = True

        return longs, shorts
