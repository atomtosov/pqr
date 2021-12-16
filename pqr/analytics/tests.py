from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Literal, Sequence, Protocol, Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from pqr.core import (
    Portfolio, Allocator, Universe,
    Factor, Filter, LookBackPctChange, Lag, Hold, ReplaceWithNan)
from pqr.utils import align_many
from .metrics import NumericMetric, TimeSeriesMetric

__all__ = [
    "Test",
    "TTest",
    "MonkeyTest",
    "OpportunitiesTest",
]


class Test(Protocol):
    def test(self, portfolio: Portfolio) -> Any:
        pass


@dataclass
class TTest:
    h0: float = 0.0
    alternative: str = "greater"

    def test(self, portfolio: Portfolio):
        return ttest_1samp(portfolio.returns, self.h0, alternative=self.alternative)


@dataclass
class OpportunitiesTest:
    universe: Universe
    allocation_strategy: Allocator | Sequence[Allocator]
    performance_metric: NumericMetric | TimeSeriesMetric
    holding: Optional[int] = None

    def test(self, portfolio: Portfolio) -> pd.Series:
        data_to_test = self._prepare_data_to_test(portfolio)
        best_portfolio = self._get_best_portfolio(*data_to_test)
        worst_portfolio = self._get_worst_portfolio(*data_to_test)

        if isinstance(self.performance_metric, NumericMetric):
            performance_metric = self.performance_metric.trailing
        else:
            performance_metric = self.performance_metric.calculate

        best_metric = performance_metric(best_portfolio)
        worst_metric = performance_metric(worst_portfolio)
        portfolio_metric = performance_metric(portfolio)
        best_metric, worst_metric, portfolio_metric = align_many(
            best_metric, worst_metric, portfolio_metric)

        return (portfolio_metric - worst_metric) / (best_metric - worst_metric)

    def get_best_portfolio(self, portfolio: Portfolio) -> Portfolio:
        return self._get_best_portfolio(*self._prepare_data_to_test(portfolio))

    def get_worst_portfolio(self, portfolio: Portfolio) -> Portfolio:
        return self._get_worst_portfolio(*self._prepare_data_to_test(portfolio))

    def _get_best_portfolio(
            self,
            long_limits: pd.DataFrame,
            short_limits: pd.DataFrame,
            universe: Universe,
            holding: int
    ) -> Portfolio:
        best_portfolio = Portfolio(
            *self._pick_with_forward_looking(
                long_limits, short_limits,
                universe, holding, "best"),
            name="Best")
        best_portfolio.allocate(self.allocation_strategy)
        best_portfolio.calculate_returns(self.universe)
        return best_portfolio

    def _get_worst_portfolio(
            self,
            long_limits: pd.DataFrame,
            short_limits: pd.DataFrame,
            universe: Universe,
            holding: int
    ) -> Portfolio:
        worst_portfolio = Portfolio(
            *self._pick_with_forward_looking(
                long_limits, short_limits,
                universe, holding, "worst"),
            name="Worst")
        worst_portfolio.allocate(self.allocation_strategy)
        worst_portfolio.calculate_returns(self.universe)
        return worst_portfolio

    def _prepare_data_to_test(self, portfolio: Portfolio):
        longs, shorts = portfolio.get_longs(), portfolio.get_shorts()
        longs, shorts, prices, mask = align_many(
            longs, shorts,
            self.universe.prices, self.universe.mask)

        long_limits = longs.sum(axis=1)
        short_limits = shorts.sum(axis=1)
        universe = Universe(prices)
        universe.filter(mask)

        if self.holding is None:
            holding = estimate_holding(portfolio.picks)
        else:
            holding = self.holding

        return long_limits, short_limits, universe, holding

    @staticmethod
    def _pick_with_forward_looking(
            long_limits: pd.DataFrame,
            short_limits: pd.DataFrame,
            universe: Universe,
            holding: int,
            how: Literal["best", "worst"]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        returns = Factor(
            universe.prices,
            better="more",
            preprocessor=[
                Filter(universe.mask),
                LookBackPctChange(holding),
                Lag(-holding),
                Hold(holding),
                ReplaceWithNan(0),
            ]
        )
        universe_returns = returns.values

        longs = pd.DataFrame(
            np.zeros_like(universe_returns, dtype=bool),
            index=universe_returns.index.copy(),
            columns=universe_returns.columns.copy(),
        )
        shorts = pd.DataFrame(
            np.zeros_like(universe_returns, dtype=bool),
            index=universe_returns.index.copy(),
            columns=universe_returns.columns.copy()
        )

        for date in universe_returns.index:
            if how == "best":
                best = universe_returns.loc[date].nlargest(long_limits.loc[date]).index
                worst = universe_returns.loc[date].nsmallest(short_limits.loc[date]).index
            else:
                best = universe_returns.loc[date].nsmallest(long_limits.loc[date]).index
                worst = universe_returns.loc[date].nlargest(short_limits.loc[date]).index

            longs.loc[date, best] = True
            shorts.loc[date, worst] = True

        return longs, shorts


@dataclass
class MonkeyTest:
    universe: Universe
    allocation_strategy: Allocator | Sequence[Allocator]
    target_metric: Callable[[Portfolio], pd.Series]
    holding: Optional[int] = None
    n: int = 100
    seed: Optional[int] = None

    def test(self, portfolio: Portfolio) -> pd.Series:
        longs, shorts = portfolio.get_longs(), portfolio.get_shorts()
        universe, longs, shorts = align_many(self.universe.mask, longs, shorts)

        long_limits = np.nansum(longs.to_numpy(), axis=1)
        short_limits = np.nansum(shorts.to_numpy(), axis=1)

        rng = np.random.default_rng(self.seed)
        if self.holding is None:
            holding = estimate_holding(portfolio.picks)
        else:
            holding = self.holding

        random_metrics = []
        for _ in range(self.n):
            random_portfolio = Portfolio(
                *self._pick_randomly(universe, long_limits, short_limits, rng, holding)
            )
            random_portfolio.allocate(self.allocation_strategy)
            random_portfolio.calculate_returns(self.universe)

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

    @staticmethod
    def _pick_randomly(
            universe: pd.DataFrame,
            long_limits: np.ndarray,
            short_limits: np.ndarray,
            rng: np.random.Generator,
            holding: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        universe_arr = universe.to_numpy()
        longs = np.zeros_like(universe, dtype=bool)
        shorts = np.zeros_like(universe, dtype=bool)

        for i in range(0, len(universe_arr), holding):
            available = np.where(universe_arr[i])[0]
            n_longs, n_shorts = long_limits[i], short_limits[i]

            choice = rng.choice(
                available,
                size=n_longs + n_shorts,
                replace=False
            )

            hold = min(len(universe_arr), i + holding)
            longs[i:hold, choice[:n_longs]] = True
            shorts[i:hold, choice[n_longs:]] = True

        longs = pd.DataFrame(
            longs,
            index=universe.index.copy(),
            columns=universe.columns.copy()
        )
        shorts = pd.DataFrame(
            shorts,
            index=universe.index.copy(),
            columns=universe.columns.copy()
        )

        return longs, shorts


def estimate_holding(picks: pd.DataFrame) -> int:
    diff = np.diff(picks.to_numpy(), axis=0)
    rebalancings_long = (diff == 1).any(axis=1).sum()
    rebalancings_short = (diff == -1).any(axis=1).sum()
    avg_rebalacings = (rebalancings_long + rebalancings_short) / 2

    return round(len(diff) / avg_rebalacings)
