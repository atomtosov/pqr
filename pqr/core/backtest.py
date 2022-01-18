__all__ = [
    "backtest_portfolio",
    "backtest_factor_portfolios",
    "grid_search_factor_portfolios",
]

from typing import (
    Sequence,
    Optional,
    Callable,
    List,
    Dict,
)

import pandas as pd

from pqr.core.allocation import equal_weights
from pqr.core.picking import pick
from pqr.core.returns import calculate_returns


def backtest_portfolio(
        prices: pd.DataFrame,
        longs: Optional[pd.DataFrame] = None,
        shorts: Optional[pd.DataFrame] = None,
        allocation: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        name: Optional[str] = None,
) -> pd.DataFrame:
    picks = pick(longs, shorts)
    if allocation is None:
        holdings = equal_weights(picks)
    else:
        holdings = allocation(picks)
    portfolio = calculate_returns(prices, holdings)

    portfolio.index.name = name or "Portfolio"

    return portfolio


def backtest_factor_portfolios(
        factor: pd.DataFrame,
        prices: pd.DataFrame,
        strategies: Sequence[Callable[[pd.DataFrame], pd.DataFrame]],
        allocation: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        add_wml: bool = False,
) -> List[pd.DataFrame]:
    names = (["Winners"] +
             [f"Neutral {i}" for i in range(1, len(strategies) - 1)] +
             ["Losers"])

    portfolios = []
    for name, strategy in zip(names, strategies):
        portfolios.append(
            backtest_portfolio(
                prices=prices,
                longs=strategy(factor),
                allocation=allocation,
                name=name,
            )
        )

    if add_wml:
        portfolios.append(
            backtest_portfolio(
                prices=prices,
                longs=strategies[0](factor),
                shorts=strategies[1](factor),
                allocation=allocation,
                name="WML",
            )
        )

    return portfolios


def grid_search_factor_portfolios(
        factor: pd.DataFrame,
        transforms: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]],
        metric: Callable[[pd.DataFrame], float],
        *args,
        **kwargs,
) -> pd.DataFrame:
    metrics_grid = []

    for name, transform in transforms.items():
        metrics_grid.append(
            pd.Series({
                portfolio.name: metric(portfolio)
                for portfolio in backtest_factor_portfolios(factor, *args, **kwargs)
            }, name=name)
        )

    return pd.DataFrame(metrics_grid)
