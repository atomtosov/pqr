from __future__ import annotations

__all__ = [
    "Portfolio",
]

from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import pandas as pd

from pqr.core.allocation import equal_weights
from pqr.utils import align


@dataclass
class Portfolio:
    holdings: pd.DataFrame = field(repr=False)
    returns: pd.Series = field(repr=False)
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.name is not None:
            self.holdings.index.name = self.name
            self.returns.index.name = self.name

    @classmethod
    def backtest(
            cls,
            calculator: Callable[[pd.DataFrame], pd.Series],
            allocator: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
            longs: Optional[pd.DataFrame] = None,
            shorts: Optional[pd.DataFrame] = None,
            name: Optional[str] = None,
    ) -> Portfolio:
        if longs is None and shorts is None:
            raise ValueError("either longs or shorts must be given")
        elif longs is not None and shorts is not None:
            longs, shorts = align(longs, shorts)
            picks = longs.astype(np.int8) - shorts.astype(np.int8)
        elif longs is not None and shorts is None:
            picks = longs.astype(np.int8)
        else:
            picks = shorts.astype(np.int8)

        if allocator is None:
            allocator = equal_weights

        holdings = allocator(picks)
        returns = calculator(holdings)
        return cls(
            holdings=holdings,
            returns=returns,
            name=name,
        )

    @classmethod
    def from_pandas(cls, portfolio: pd.DataFrame) -> Portfolio:
        return cls(
            holdings=portfolio.drop(columns="returns"),
            returns=portfolio["returns"],
            name=portfolio.index.name,
        )

    def to_pandas(self) -> pd.DataFrame:
        holdings, returns = align(self.holdings, self.returns)
        return pd.concat([holdings, returns], axis=1)

    def get_long_picks(self) -> pd.DataFrame:
        return self.holdings > 0

    def get_short_picks(self) -> pd.DataFrame:
        return self.holdings < 0

    def get_long_holdings(self) -> pd.DataFrame:
        return self.holdings[self.holdings > 0].fillna(0)

    def get_short_holdings(self) -> pd.DataFrame:
        return self.holdings[self.holdings < 0].fillna(0)
