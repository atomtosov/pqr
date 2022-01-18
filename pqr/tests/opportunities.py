from typing import (
    Optional,
    Callable,
)

import numpy as np
import pandas as pd

from pqr.core import (
    backtest_portfolio,
    filter,
    look_back_pct_change,
    lag,
    hold,
)
from pqr.utils import (
    estimate_holding,
    align,
    compose,
    partial,
    longs_from_portfolio,
    shorts_from_portfolio,
)


def opportunities_test(
        portfolio: pd.DataFrame,
        prices: pd.DataFrame,
        universe: Optional[pd.DataFrame] = None,
        allocation: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        holding: Optional[int] = None,
) -> pd.Series:
    if universe is None:
        universe = prices.notnull()

    if holding is None:
        holding = estimate_holding(portfolio)

    best_portfolio = get_best_portfolio(
        portfolio=portfolio,
        prices=prices,
        universe=universe,
        allocation=allocation,
        holding=holding,
    )
    worst_portfolio = get_worst_portfolio(
        portfolio=portfolio,
        prices=prices,
        universe=universe,
        allocation=allocation,
        holding=holding,
    )

    op_est = (
            (best_portfolio["returns"] - portfolio["returns"]) /
            (best_portfolio["returns"] - worst_portfolio["returns"])
    ).dropna()
    op_est.index.name = "Opportunities"

    return op_est


def get_best_portfolio(
        portfolio: pd.DataFrame,
        prices: pd.DataFrame,
        universe: Optional[pd.DataFrame] = None,
        allocation: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        holding: Optional[int] = None,
) -> pd.DataFrame:
    if universe is None:
        universe = prices.notnull()

    if holding is None:
        holding = estimate_holding(portfolio)

    return _get_extreme_portfolio(
        base_portfolio=portfolio,
        prices=prices,
        universe=universe,
        allocation=allocation,
        holding=holding,
        how="best",
    )


def get_worst_portfolio(
        portfolio: pd.DataFrame,
        prices: pd.DataFrame,
        universe: Optional[pd.DataFrame] = None,
        allocation: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        holding: Optional[int] = None
) -> pd.DataFrame:
    if universe is None:
        universe = prices.notnull()

    if holding is None:
        holding = estimate_holding(portfolio)

    return _get_extreme_portfolio(
        base_portfolio=portfolio,
        prices=prices,
        universe=universe,
        allocation=allocation,
        holding=holding,
        how="worst",
    )


def _get_extreme_portfolio(
        base_portfolio: pd.DataFrame,
        prices: pd.DataFrame,
        universe: pd.DataFrame,
        allocation: Callable[[pd.DataFrame], pd.DataFrame],
        holding: int,
        how: str
) -> pd.DataFrame:
    longs, shorts = longs_from_portfolio(base_portfolio), shorts_from_portfolio(base_portfolio)

    if how == "best":
        how_long = "best"
        how_short = "worst"
    else:
        how_long = "worst"
        how_short = "best"

    return backtest_portfolio(
        prices=prices,
        longs=pick_with_forward_looking(
            prices=prices,
            universe=universe,
            holdings_number=longs.sum(axis=1),
            holding=holding,
            how=how_long,
        ),
        shorts=pick_with_forward_looking(
            prices=prices,
            universe=universe,
            holdings_number=shorts.sum(axis=1),
            holding=holding,
            how=how_short,
        ),
        allocation=allocation,
        name=how.capitalize(),
    )


def pick_with_forward_looking(
        prices: pd.DataFrame,
        universe: pd.DataFrame,
        holdings_number: pd.Series,
        holding: Optional[int] = None,
        how: str = "best",
) -> pd.DataFrame:
    forward_looking_transform = compose(
        partial(filter, universe=universe),
        partial(look_back_pct_change, period=holding),
        partial(lag, period=-holding),
        partial(hold, period=holding),
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
    stop = len(choices)
    for i in range(0, stop, holding):
        choice = choosing_func(
            forward_returns_array[i],
            holdings_number_array[i],
        )
        choices[i:min(stop, i + holding), choice] = True

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
        return np.where(arr >= sorted_values[-n])[0]

    return np.array([], dtype=int)


def _bottom_idx(arr: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)

    unique_values = np.unique(arr[~np.isnan(arr)])
    if unique_values.any():
        sorted_values = np.sort(unique_values)
        return np.where(arr < sorted_values[n])[0]

    return np.array([], dtype=int)
