from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from pqr.utils import align

__all__ = [
    "Filter",
    "LookBack",
    "Lag",
    "Hold",
]

AggregationFunction = Callable[[pd.DataFrame], pd.Series]


@dataclass
class Filter:
    universe: pd.DataFrame = field(repr=False)

    def __call__(self, values: pd.DataFrame) -> pd.DataFrame:
        universe, values = align(self.universe, values)
        return pd.DataFrame(
            np.where(universe.to_numpy(), values.to_numpy(), np.nan),
            index=values.index,
            columns=values.columns
        )


@dataclass
class LookBack:
    agg_func: AggregationFunction
    period: int

    def __call__(self, values: pd.DataFrame) -> pd.DataFrame:
        rows = []
        window = self.period + 1
        for i in range(window - 1, len(values)):
            rows.append(
                self.agg_func(values.iloc[i - window + 1:i + 1])
            )

        return pd.DataFrame(rows, index=values.index[window - 1:])


@dataclass
class Lag:
    period: int

    def __call__(self, values: pd.DataFrame) -> pd.DataFrame:
        if self.period == 0:
            return values

        return pd.DataFrame(
            values.to_numpy()[:-self.period],
            index=values.index[self.period:],
            columns=values.columns
        )


@dataclass
class Hold:
    period: int

    def __call__(self, values: pd.DataFrame) -> pd.DataFrame:
        periods = np.zeros(len(values), dtype=int)
        update_periods = np.arange(len(values), step=self.period)
        periods[update_periods] = update_periods
        update_mask = np.maximum.accumulate(periods[:, np.newaxis], axis=0)

        return pd.DataFrame(
            np.take_along_axis(values.to_numpy(), update_mask, axis=0),
            index=values.index,
            columns=values.columns
        )
