from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .portfolio import Portfolio, Allocator
from .universe import Universe

__all__ = [
    "Benchmark",
]


@dataclass
class Benchmark:
    returns: pd.Series = field(repr=False)
    name: Optional[str] = None

    def __post_init__(self):
        self.returns = self.returns.astype(float)
        if not self.name:
            self.name = "Benchmark"

        self.returns.index.name = self.name

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
            allocation_strategy: Optional[Allocator] = None,
            name: Optional[str] = None,
    ) -> Benchmark:
        benchmark = Portfolio(longs=universe.mask)
        benchmark.allocate(allocation_strategy)
        benchmark.calculate_returns(universe)

        return cls(
            benchmark.returns,
            name
        )
