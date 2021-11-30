from __future__ import annotations

from typing import Optional, Callable

import pandas as pd

from .portfolio import Portfolio, AllocationStep
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
            index.pct_change().fillna(0),
            name
        )

    @classmethod
    def from_universe(
            cls,
            universe: Universe,
            weighting_strategy: Optional[AllocationStep] = None,
            name: Optional[str] = None,
    ) -> Benchmark:
        benchmark = Portfolio(
            universe,
            longs=universe.mask,
            allocation_strategy=weighting_strategy
        )

        return cls(
            benchmark.returns,
            name
        )
