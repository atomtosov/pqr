from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils import array_to_alike_df_or_series, trail

__all__ = [
    "Factor",
]


class Factor:
    __slots__ = (
        "values",
    )

    def __init__(self, values: pd.DataFrame | pd.Series):
        self.values = values.astype(float)

    def look_back(
            self,
            agg: Callable[[np.ndarray], npt.ArrayLike],
            period: int,
            **kwargs,
    ) -> Factor:
        self.values: pd.DataFrame | pd.Series = trail(
            self.values,
            func=agg,
            window=period + 1,
            **kwargs
        )

        return self

    def lag(self, period: int) -> Factor:
        self.values: pd.DataFrame | pd.Series = array_to_alike_df_or_series(
            self.values.to_numpy()[period:],
            self.values.iloc[period:]
        )

        return self

    def hold(self, period: int) -> Factor:
        periods: np.ndarray = np.zeros(len(self.values), dtype=int)
        update_periods: np.ndarray = np.arange(len(self.values), step=period)
        periods[update_periods]: np.ndarray = update_periods
        update_mask: np.ndarray = np.maximum.accumulate(periods[:, np.newaxis], axis=0)

        self.values: pd.DataFrame | pd.Series = array_to_alike_df_or_series(
            np.take_along_axis(self.values.to_numpy(), update_mask, axis=0),
            self.values
        )

        return self

    def quantile(self, q: float) -> pd.DataFrame:
        return array_to_alike_df_or_series(
            np.nanquantile(
                self.values.to_numpy(), q, axis=1
            ),
            self.values
        ).rename(index=f"q_{q:.2f}")

    def top(self, place: int) -> pd.DataFrame:
        # TODO: optimize with numpy
        return self.values.apply(
            lambda row: pd.Series.nlargest(row, place).min(), axis=1
        ).rename(index=f"top_{place:.2f}")

    def bottom(self, place: int) -> pd.DataFrame:
        return self.values.apply(
            lambda row: pd.Series.nsmallest(row, place).max(), axis=1
        ).rename(index=f"bottom_{place:.2f}")
