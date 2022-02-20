from __future__ import annotations

__all__ = [
    "Benchmark",
]

from dataclasses import dataclass, field
from typing import (
    Optional,
    Callable,
    Any,
)

import pandas as pd

from pqr.core.portfolio import Portfolio


@dataclass
class Benchmark:
    returns: pd.Series = field(repr=False)
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.name is not None:
            self.returns.index.name = self.name

    @classmethod
    def from_index(
            cls,
            index: pd.Series,
            name: Optional[str],
    ) -> Benchmark:
        return cls(
            returns=index.pct_change().fillna(0),
            name=name,
        )

    @classmethod
    def from_portfolio(
            cls,
            portfolio: Portfolio,
            name: Optional[str] = None,
    ) -> Benchmark:
        return cls(
            returns=portfolio.returns,
            name=name or portfolio.name,
        )

    @classmethod
    def from_universe(
            cls,
            universe: pd.DataFrame,
            allocator: Callable[[pd.DataFrame], pd.DataFrame],
            calculator: Callable[[pd.DataFrame], pd.Series],
            name: Optional[str] = None,
    ) -> Benchmark:
        return cls.from_portfolio(
            portfolio=Portfolio.backtest(
                longs=universe,
                shorts=None,
                allocator=allocator,
                calculator=calculator,
                name=name or "Benchmark",
            ),
            name=name or "Benchmark",
        )

    def starting_from(self, idx: Any) -> Benchmark:
        returns = self.returns.copy()[idx:]
        returns.iat[0] = 0
        return Benchmark(
            returns=returns,
            name=self.name,
        )

    def to_pandas(self) -> pd.Series:
        return self.returns.copy()
