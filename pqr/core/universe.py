from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Universe:
    prices: pd.DataFrame = field(repr=False)
    mask: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        self.prices = self.prices.astype(float)
        self.mask = self.prices.notnull()

    def filter(self, mask: pd.DataFrame):
        self.mask &= mask

    def describe(self) -> pd.Series:
        pass
