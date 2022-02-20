from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import pandas as pd

from pqr.core import Portfolio
from pqr.factors import filter, look_back_pct_change, lag
from pqr.utils import align, compose, partial


def opportunities_test(
        base_portfolio: Portfolio,
        prices: pd.DataFrame,
        universe: pd.DataFrame,
        allocator: Callable[[pd.DataFrame], pd.DataFrame],
        calculator: Callable[[pd.DataFrame], pd.Series],
) -> pd.Series:
    best_portfolio = get_extreme_portfolio(
        base_portfolio=base_portfolio,
        prices=prices,
        universe=universe,
        allocator=allocator,
        calculator=calculator,
        how="best",
    )
    worst_portfolio = get_extreme_portfolio(
        base_portfolio=base_portfolio,
        prices=prices,
        universe=universe,
        allocator=allocator,
        calculator=calculator,
        how="worst",
    )

    best_portfolio_returns, worst_portfolio_returns, base_portfolio_returns = align(
        best_portfolio.returns,
        worst_portfolio.returns.iloc[1:],
        base_portfolio.returns.iloc[1:],
    )
    op_est = (
            (best_portfolio_returns - base_portfolio_returns) /
            (best_portfolio_returns - worst_portfolio_returns)
    ).dropna()
    op_est.name = "Opportunities"

    return op_est


def get_extreme_portfolio(
        base_portfolio: Portfolio,
        prices: pd.DataFrame,
        universe: pd.DataFrame,
        allocator: Callable[[pd.DataFrame], pd.DataFrame],
        calculator: Callable[[pd.DataFrame], pd.Series],
        how: Literal["best", "worst"],
) -> Portfolio:
    longs = base_portfolio.get_long_picks()
    shorts = base_portfolio.get_short_picks()

    if how == "best":
        how_long = "best"
        how_short = "worst"
    else:
        how_long = "worst"
        how_short = "best"

    return Portfolio.backtest(
        longs=pick_with_forward_looking(
            prices=prices,
            universe=universe,
            holdings_number=longs.sum(axis=1),
            how=how_long,
        ),
        shorts=pick_with_forward_looking(
            prices=prices,
            universe=universe,
            holdings_number=shorts.sum(axis=1),
            how=how_short,
        ),
        allocator=allocator,
        calculator=calculator,
        name=how.capitalize(),
    )


def pick_with_forward_looking(
        prices: pd.DataFrame,
        universe: pd.DataFrame,
        holdings_number: pd.Series,
        how: str,
) -> pd.DataFrame:
    prices, universe, holdings_number = align(
        prices,
        universe,
        holdings_number,
    )
    forward_looking_transform = compose(
        partial(filter, universe=universe),
        partial(look_back_pct_change, period=1),
        partial(lag, period=-1),
    )
    forward_returns, holdings_number = align(
        forward_looking_transform(prices),
        holdings_number,
    )
    forward_returns_array = forward_returns.to_numpy()
    holdings_number_array = holdings_number.to_numpy()

    if how == "best":
        choosing_func = _top_idx
    else:
        choosing_func = _bottom_idx

    choices = np.zeros_like(forward_returns, dtype=bool)
    for i in range(0, len(choices)):
        choice = choosing_func(
            forward_returns_array[i],
            holdings_number_array[i],
        )
        choices[i, choice] = True

    return pd.DataFrame(
        choices,
        index=forward_returns.index.copy(),
        columns=forward_returns.columns.copy(),
    )


def _top_idx(arr: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)

    unique_values = np.unique(arr[~np.isnan(arr)])
    if unique_values.any():
        sorted_values = np.sort(unique_values)
        return np.where(arr >= sorted_values[-min(n, len(sorted_values))])[0][:n]

    return np.array([], dtype=int)


def _bottom_idx(arr: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)

    unique_values = np.unique(arr[~np.isnan(arr)])
    if unique_values.any():
        sorted_values = np.sort(unique_values)
        return np.where(arr <= sorted_values[min(n, len(sorted_values)) - 1])[0][:n]

    return np.array([], dtype=int)
