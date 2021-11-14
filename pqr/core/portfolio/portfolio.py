from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

__all__ = [
    "Portfolio",
]


@dataclass
class Portfolio:
    picks: pd.DataFrame = field(repr=False)
    name: Optional[str] = None

    weights: pd.DataFrame = field(init=False, repr=False)
    positions: pd.DataFrame = field(init=False, repr=False)
    returns: pd.Series = field(init=False, repr=False)

    def __post_init__(self):
        if self.name is None:
            self.name = "Portfolio"

        self.picks.index.name = self.name

    def set_picks(self, picks: pd.DataFrame) -> None:
        self.picks = picks
        self.picks.index.name = self.name

    def set_weights(self, weights: pd.DataFrame) -> None:
        self.weights = weights
        self.weights.index.name = self.name

    def set_positions(self, positions: pd.DataFrame) -> None:
        self.positions = positions
        self.positions.index.name = self.name

    def set_returns(self, returns: pd.Series) -> None:
        self.returns = returns
        self.returns.name = self.name

    def get_longs(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.picks.to_numpy() == 1,
            index=self.picks.index.copy(),
            columns=self.picks.columns.copy()
        )

    def get_shorts(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.picks.to_numpy() == -1,
            index=self.picks.index.copy(),
            columns=self.picks.columns.copy()
        )
