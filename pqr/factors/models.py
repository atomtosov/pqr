from __future__ import annotations

__all__ = [
    "backtest_factor_portfolios",
    "grid_search_factor_portfolios",
]

from typing import (
    Optional,
    Callable,
)

import pandas as pd

from pqr.core import Portfolio


def backtest_factor_portfolios(
        factor: pd.DataFrame,
        strategies: dict[str, Callable[[pd.DataFrame], pd.DataFrame]],
        allocator: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
        calculator: Callable[[pd.DataFrame], pd.Series],
        add_wml: bool = False,
) -> list[Portfolio]:
    portfolios = []
    for name, strategy in strategies.items():
        portfolios.append(
            Portfolio.backtest(
                calculator=calculator,
                longs=strategy(factor),
                shorts=None,
                allocator=allocator,
                name=name,
            )
        )

    if add_wml:
        portfolios.append(
            Portfolio.backtest(
                calculator=calculator,
                longs=portfolios[0].get_long_picks(),
                shorts=portfolios[-1].get_long_picks(),
                allocator=allocator,
                name="WML",
            )
        )

    return portfolios


def grid_search_factor_portfolios(
        factor: pd.DataFrame,
        transforms: dict[str, Callable[[pd.DataFrame], pd.DataFrame]],
        metric: Callable[[Portfolio], float],
        *args,
        **kwargs,
) -> pd.DataFrame:
    metrics_grid = []

    for name, transform in transforms.items():
        metrics_grid.append(
            pd.Series({
                portfolio.name: metric(portfolio)
                for portfolio in backtest_factor_portfolios(
                    transform(factor),
                    *args, **kwargs
                )
            }, name=name)
        )

    return pd.DataFrame(metrics_grid)
