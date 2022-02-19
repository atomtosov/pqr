from __future__ import annotations

__all__ = [
    "calculate_returns",
    "prices_to_returns",
]

import numpy as np
import pandas as pd

from pqr.utils import align


def calculate_returns(
        holdings: pd.DataFrame,
        universe_returns: pd.DataFrame,
) -> pd.Series:
    universe_returns, holdings = align(
        universe_returns,
        holdings
    )

    portfolio_returns = universe_returns.to_numpy()[1:] * holdings.to_numpy()[:-1]
    returns = np.insert(
        np.nansum(portfolio_returns, axis=1),
        obj=0, values=0,
    )

    return pd.Series(
        returns,
        index=holdings.index.copy(),
    )


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices_array = prices.to_numpy(float)
    universe_returns = np.diff(prices_array, axis=0) / prices_array[:-1]

    # nan means that returns are unknown, fix them to zero
    universe_returns = np.nan_to_num(
        universe_returns,
        nan=0, neginf=0, posinf=0,
    )

    # 1st period returns are unknown, again fix them as zero
    universe_returns = np.insert(
        universe_returns, 0,
        values=0, axis=0,
    )

    return pd.DataFrame(
        universe_returns,
        index=prices.index.copy(),
        columns=prices.columns.copy(),
    )
