from __future__ import annotations

from typing import Callable, Optional, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

import pqr
from pqr.core import Portfolio, AllocationStep, Universe, Hold
from pqr.utils import align_many
from .utils import estimate_holding

__all__ = [
    "TTest",
    "ZeroIntelligenceTest",
    "OpportunitiesTest"
]


class TTest:
    def __init__(
            self,
            h0: float = 0.0,
            alternative: str = "greater"
    ):
        self.h0 = h0
        self.alternative = alternative

    def __call__(self, portfolio: Portfolio):
        return ttest_1samp(portfolio.returns, self.h0, alternative=self.alternative)


class ZeroIntelligenceTest:
    def __init__(
            self,
            universe: Universe,
            allocation_strategy: AllocationStep | Sequence[AllocationStep],
            target_metric: Callable[[Portfolio], pd.Series],
            holding: Optional[int] = None,
            n: int = 100,
            seed: Optional[int] = None,
    ):
        self.universe = universe
        self.allocation_strategy = allocation_strategy
        self.target_metric = target_metric
        self.n = n
        self.holding = holding
        self.seed = seed

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        random_picking = RandomPicking(
            self.universe,
            portfolio,
            self.n,
            self.holding,
            np.random.default_rng(self.seed)
        )

        random_metrics = []
        for longs, shorts in random_picking:
            random_portfolio = Portfolio(
                self.universe,
                longs=longs,
                shorts=shorts,
                allocation_strategy=self.allocation_strategy,
            )

            random_metrics.append(
                self.target_metric(random_portfolio).to_numpy()
            )

        random_metrics = np.array(random_metrics).T
        portfolio_metric = self.target_metric(portfolio)

        return pd.Series(
            np.nanmean(
                random_metrics <= portfolio_metric.to_numpy()[:, np.newaxis],
                axis=1
            ),
            index=portfolio_metric.index
        )


class RandomPicking:
    def __init__(
            self,
            universe: Universe,
            portfolio: Portfolio,
            n: int = 100,
            holding: Optional[int] = None,
            rng: Optional[np.random.Generator] = None,
    ):
        self.portfolio = portfolio
        self.n = n
        self.holding = holding if holding is not None else estimate_holding(portfolio.picks)
        self.rng = rng if rng is not None else np.random.default_rng()

        longs, shorts = portfolio.get_longs(), portfolio.get_shorts()
        universe, longs, shorts = align_many(universe.mask, longs, shorts)
        self._universe = universe.to_numpy()
        self._long_limits = np.nansum(longs.to_numpy(), axis=1)
        self._short_limits = np.nansum(shorts.to_numpy(), axis=1)

        self._i = 0

    def __iter__(self) -> RandomPicking:
        return self

    def __next__(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._i < self.n:
            self._i += 1
            return self._generate_random_picks()

        raise StopIteration

    def __len__(self) -> int:
        return self.n

    def _generate_random_picks(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        longs = np.zeros_like(self._universe, dtype=bool)
        shorts = np.zeros_like(self._universe, dtype=bool)

        for i in range(0, len(self._universe), self.holding):
            available = np.where(self._universe[i])[0]
            n_longs, n_shorts = self._long_limits[i], self._short_limits[i]

            choice = self.rng.choice(
                available,
                size=n_longs + n_shorts,
                replace=False
            )

            hold = min(len(self._universe), i + self.holding)
            longs[i:hold, choice[:n_longs]] = True
            shorts[i:hold, choice[n_longs:]] = True

        longs = pd.DataFrame(
            longs,
            index=self.portfolio.picks.index,
            columns=self.portfolio.picks.columns
        )
        shorts = pd.DataFrame(
            shorts,
            index=self.portfolio.picks.index,
            columns=self.portfolio.picks.columns
        )

        return longs, shorts


class OpportunitiesTest:
    def __init__(
            self,
            universe: Universe,
            allocation_strategy: AllocationStep | Sequence[AllocationStep],
            target_metric: Callable[[Portfolio], pd.Series],
            holding: Optional[int] = None
    ):
        self.universe = universe
        self.allocation_strategy = allocation_strategy
        self.target_metric = target_metric
        self.holding = holding

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        holding = estimate_holding(portfolio.picks) if self.holding is None else self.holding

        best_portfolio = pqr.Portfolio(
            self.universe,
            *ProphetPickingStrategy(self.universe, holding, "best")(portfolio),
            allocation_strategy=self.allocation_strategy
        )
        worst_portfolio = pqr.Portfolio(
            self.universe,
            *ProphetPickingStrategy(self.universe, holding, "worst")(portfolio),
            allocation_strategy=self.allocation_strategy
        )

        best_metric = self.target_metric(best_portfolio)
        worst_metric = self.target_metric(worst_portfolio)
        portfolio_metric = self.target_metric(portfolio)

        return (portfolio_metric - worst_metric) / (best_metric - worst_metric)


class ProphetPickingStrategy:
    def __init__(
            self,
            universe: Universe,
            holding: int,
            way: Literal["best", "worst"]
    ):
        self.universe = universe
        self.holding = holding
        self.way = way

    def __call__(self, portfolio: Portfolio) -> tuple[pd.DataFrame, pd.DataFrame]:
        longs, shorts = portfolio.get_longs(), portfolio.get_shorts()
        longs, shorts, universe, prices = align_many(longs, shorts, self.universe.mask, self.universe.prices)

        long_limits = longs.sum(axis=1)
        short_limits = shorts.sum(axis=1)

        universe_returns = prices.pct_change(self.holding).shift(-self.holding)
        universe_returns = Hold(self.holding)(universe_returns)
        universe_returns[~universe] = np.nan

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
            if self.way == "best":
                best = universe_returns.loc[date].nlargest(long_limits.loc[date]).index
                worst = universe_returns.loc[date].nsmallest(short_limits.loc[date]).index
            else:
                best = universe_returns.loc[date].nsmallest(long_limits.loc[date]).index
                worst = universe_returns.loc[date].nlargest(short_limits.loc[date]).index

            longs.loc[date, best] = True
            shorts.loc[date, worst] = True

        return longs, shorts
