from __future__ import annotations

from typing import Optional

import pandas as pd

from .factor import Factor
from .portfolio import Portfolio
from .universe import Universe

__all__ = [
    "Benchmark",
]


class Benchmark:
    __slots__ = (
        "name",
        "returns",
    )

    def __init__(
            self,
            returns: pd.Series,
            name: Optional[str] = None,
    ):
        self.returns = returns.astype(float)

        if name:
            self.name = name
        elif self.returns.index.name:
            self.name = self.returns.index.name
        else:
            self.name = "benchmark"

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
            universe: Optional[Universe] = None,
            weighting_factor: Optional[Factor] = None,
            name: Optional[str] = None,
    ) -> Benchmark:
        if universe is None:
            universe: Universe = Universe(prices)

        benchmark: Portfolio = Portfolio(name)
        benchmark.pick(universe)
        benchmark.weigh(weighting_factor)
        benchmark.allocate(prices)

        return cls(
            benchmark.returns,
            name
        )

    def __repr__(self):
        return f"Benchmark({repr(self.name)})"
