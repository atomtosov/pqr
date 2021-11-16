import pandas as pd


class Universe:
    def __init__(
            self,
            prices: pd.DataFrame
    ):
        self.prices = prices.astype(float)
        self.mask = prices.notnull()

    def filter(self, mask: pd.DataFrame):
        self.mask &= mask
