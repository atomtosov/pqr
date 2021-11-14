from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

import pandas as pd

from pqr.utils import align
from .building_steps import RelativePositions, EqualWeights
from .portfolio import Portfolio
from .portfolio_builder import PortfolioBuilder

__all__ = [
    "Benchmark",
]

WeightingStrategy = Callable[[Portfolio], Portfolio]


@dataclass
class Benchmark:
    returns: pd.Series = field(repr=False)
    name: Optional[str] = None

    def __post_init__(self):
        self.returns = self.returns.astype(float)

        if self.name is None:
            self.name = "Benchmark"

        self.returns.index.name = self.name

    @classmethod
    def from_index(
            cls,
            index: pd.Series,
            name: Optional[str] = None,
    ) -> Benchmark:
        return cls(
            index.pct_change().dropna(),
            name
        )

    @classmethod
    def from_prices(
            cls,
            prices: pd.DataFrame,
            universe: pd.DataFrame = None,
            weighting_strategy: Optional[WeightingStrategy] = None,
            name: Optional[str] = None,
    ) -> Benchmark:
        if weighting_strategy is None:
            weighting_strategy = EqualWeights()

        benchmark_builder = PortfolioBuilder(
            [
                weighting_strategy,
                RelativePositions(prices)
            ]
        )

        picks = prices.notnull()
        if universe is not None:
            picks, universe = align(picks, universe)
            picks &= universe

        benchmark = benchmark_builder(longs=picks, name=name)

        return cls(
            benchmark.returns,
            name
        )
