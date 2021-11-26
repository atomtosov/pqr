from __future__ import annotations

from typing import Optional, Callable

import pandas as pd

from .portfolio import Portfolio, PortfolioBuilder, EqualWeights, TheoreticalAllocation
from .universe import Universe

__all__ = [
    "Benchmark",
]


class Benchmark:
    def __init__(
            self,
            returns: pd.Series,
            name: Optional[str] = None
    ):
        self.returns = returns.astype(float)
        self.name = name if name is not None else "Benchmark"
        self.returns.index.name = name

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
    def from_universe(
            cls,
            universe: Universe,
            weighting_strategy: Optional[Callable[[Portfolio], Portfolio]] = None,
            name: Optional[str] = None,
    ) -> Benchmark:
        if weighting_strategy is None:
            weighting_strategy = EqualWeights()

        benchmark_builder = PortfolioBuilder(
            weighting_strategy,
            TheoreticalAllocation(),
        )

        benchmark = benchmark_builder(
            universe,
            longs=universe.mask,
            name=name
        )

        return cls(
            benchmark.returns,
            name
        )
