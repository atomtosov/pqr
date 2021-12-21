from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Universe:
    prices: pd.DataFrame = field(repr=False)
    mask: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        self.prices = self.prices.astype(float)
        self.mask = self.prices.notnull()

    def filter(self, mask: pd.DataFrame):
        # TODO: add left join for mask
        self.mask &= mask

    def get_universe_returns(self, nan_as_dead: bool = False) -> pd.DataFrame:
        prices = self.prices.to_numpy()
        universe_returns = np.diff(prices, axis=0) / prices[:-1]

        if nan_as_dead:
            nan_replacer = -1
        else:
            nan_replacer = 0
        universe_returns = np.nan_to_num(
            universe_returns,
            nan=nan_replacer, posinf=nan_replacer, neginf=nan_replacer
        )

        universe_returns = np.insert(
            universe_returns, 0,
            values=0, axis=0
        )

        return pd.DataFrame(
            universe_returns,
            index=self.prices.index.copy(),
            columns=self.prices.columns.copy()
        )

    def describe(self) -> pd.Series:
        pass
