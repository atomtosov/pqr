__all__ = [
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
                portfolio.index.name: metric(portfolio)
                for portfolio in backtest_factor_portfolios(
                    transform(factor),
                    *args, **kwargs
                )
            }, name=name)
        )

    return pd.DataFrame(metrics_grid)
