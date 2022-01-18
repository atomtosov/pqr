__all__ = [
    "calculate_returns",
]

import numpy as np
import pandas as pd

from pqr.utils import align


def calculate_returns(
        prices: pd.DataFrame,
        holdings: pd.DataFrame
) -> pd.DataFrame:
    universe_returns, holdings = align(
        _prices_to_returns(prices),
        holdings
    )

    portfolio_returns = universe_returns.to_numpy()[1:] * holdings.to_numpy()[:-1]
    returns = np.insert(
        np.nansum(portfolio_returns, axis=1),
        obj=0, values=0
    )

    holdings["returns"] = returns

    return holdings


def _prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices_array = prices.to_numpy(float)
    universe_returns = np.diff(prices_array, axis=0) / prices_array[:-1]

    # nan means that returns are unknown, fix them to zero
    universe_returns = np.nan_to_num(
        universe_returns,
        nan=0, neginf=0, posinf=0
    )

    # 1st period returns are unknown, again fix them as zero
    universe_returns = np.insert(
        universe_returns, 0,
        values=0, axis=0
    )

    return pd.DataFrame(
        universe_returns,
        index=prices.index.copy(),
        columns=prices.columns.copy()
    )
