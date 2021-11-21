import pandas as pd

__all__ = [
    "PctChange",
    "Mean",
    "Median",
]


class PctChange:
    def __init__(self, period: int):
        self.period = period

    def __call__(self, values: pd.DataFrame) -> pd.DataFrame:
        return values.pct_change(self.period).iloc[self.period:]


class Mean:
    def __init__(self, period: int):
        self.period = period

    def __call__(self, values: pd.DataFrame) -> pd.DataFrame:
        return values.rolling(self.period + 1, axis=0).mean().iloc[self.period:]


class Median:
    def __init__(self, period: int):
        self.period = period

    def __call__(self, values: pd.DataFrame) -> pd.DataFrame:
        return values.rolling(self.period + 1, axis=0).median().iloc[self.period:]
