import numpy as np
import pandas as pd

from pqr.utils import align_many


class Universe:
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices.astype(float)
        self.mask = prices.notnull()

    def filter(self, mask: pd.DataFrame):
        self.mask &= mask

    def __call__(self, positions: pd.DataFrame) -> pd.Series:
        prices, mask, positions = align_many(self.prices, self.mask, positions)

        universe_returns = prices.pct_change().to_numpy()[1:]

        positions_available = positions.to_numpy()[:-1]
        portfolio_returns = np.where(
            mask.to_numpy()[:-1],
            (positions_available * universe_returns), 0
        )

        dead_returns = np.where(
            np.isnan(portfolio_returns) & ~np.isclose(positions_available, 0),
            -positions_available, 0
        )
        returns = np.nansum(portfolio_returns, axis=1) + np.nansum(dead_returns, axis=1)

        return pd.Series(
            np.insert(returns, 0, 0),
            index=positions.index.copy()
        )
