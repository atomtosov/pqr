from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from pqr.core import Portfolio, PortfolioBuilder
from pqr.utils import align_many

__all__ = [
    "ZeroIntelligenceTest",
]


@dataclass
class ZeroIntelligenceTest:
    universe: pd.DataFrame
    portfolio_builder: PortfolioBuilder
    target_metric: Callable[[Portfolio], pd.Series]
    n: int = 100
    seed: Optional[int] = None

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        random_picking = RandomPicking(
            self.universe,
            portfolio,
            self.n,
            np.random.default_rng(self.seed)
        )

        random_metrics = []
        for longs, shorts in random_picking:
            random_portfolio = self.portfolio_builder(
                longs=longs,
                shorts=shorts
            )

            random_metrics.append(
                self.target_metric(random_portfolio).to_numpy()
            )

        random_metrics = np.array(random_metrics).T
        portfolio_metric = self.target_metric(portfolio)

        q = np.nanmean(
            random_metrics <= portfolio_metric.to_numpy()[:, np.newaxis],
            axis=1
        )

        return pd.Series(q, index=portfolio_metric.index)


@dataclass
class RandomPicking:
    universe: pd.DataFrame
    portfolio: Portfolio
    n: int = 100
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self):
        longs, shorts = self.portfolio.get_longs(), self.portfolio.get_shorts()

        universe, longs, shorts = align_many(self.universe, longs, shorts)

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

        for i in range(len(self._universe)):
            available = np.where(self._universe[i])[0]
            n_longs, n_shorts = self._long_limits[i], self._short_limits[i]

            choice = self.rng.choice(
                available,
                size=n_longs + n_shorts,
                replace=False
            )

            longs[i, choice[:n_longs]] = True
            shorts[i, choice[n_longs:]] = True

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
