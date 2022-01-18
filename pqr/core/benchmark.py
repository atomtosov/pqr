__all__ = [
    "index_as_benchmark",
    "universe_as_benchmark",
]

from typing import (
    Optional,
    Callable,
)

import pandas as pd

from pqr.core.backtest import backtest_portfolio


def index_as_benchmark(
        index: pd.Series,
        name: Optional[str] = None,
) -> pd.Series:
    benchmark = index.astype(float).pct_change()
    benchmark.iat[0] = 0.0
    benchmark.index.name = name or "Benchmark"
    return benchmark


def universe_as_benchmark(
        prices: pd.DataFrame,
        universe: Optional[pd.DataFrame] = None,
        allocation: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        name: Optional[str] = None,
) -> pd.Series:
    if universe is None:
        universe = prices.notnull()
    benchmark = backtest_portfolio(
        prices=prices,
        longs=universe,
        allocation=allocation,
        name=name or "Benchmark",
    )
    return benchmark["returns"]
