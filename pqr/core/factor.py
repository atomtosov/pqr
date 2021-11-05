from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .aggregations import AggregationFunction
from ..utils import array_to_alike_df_or_series, trail

__all__ = [
    "Factor",
]


class Factor:
    __slots__ = (
        "values",
    )

    def __init__(self, values: pd.DataFrame):
        if not isinstance(values, pd.DataFrame):
            raise TypeError("values of factor must be represented as pd.DataFrame")

        self.values = values.astype(float)

    def look_back(
            self,
            agg: AggregationFunction,
            period: int,
            **kwargs,
    ) -> Factor:
        self.values = trail(
            self.values,
            func=agg,
            window=period + 1,
            **kwargs
        )
        return self

    def lag(self, period: int) -> Factor:
        self.values = array_to_alike_df_or_series(
            self.values.to_numpy()[period:],
            self.values.iloc[period:]
        )
        return self

    def hold(self, period: int) -> Factor:
        periods = np.zeros(len(self.values), dtype=int)
        update_periods = np.arange(len(self.values), step=period)
        periods[update_periods] = update_periods
        update_mask = np.maximum.accumulate(periods[:, np.newaxis], axis=0)

        self.values = array_to_alike_df_or_series(
            np.take_along_axis(self.values.to_numpy(), update_mask, axis=0),
            self.values
        )

        return self

    def quantile(self, q: Sequence[float]) -> pd.DataFrame:
        quantiles = np.nanquantile(
            self.values.to_numpy(), q, axis=1
        ).T
        quantiles_labels = [f"q_{_q:.2f}" for _q in reversed(q)]

        return array_to_alike_df_or_series(
            quantiles,
            self.values.iloc[:, :len(q)]
        ).rename(columns=lambda _: quantiles_labels.pop())

    def top(self, n: Sequence[int]) -> pd.DataFrame:
        def _top(arr: np.ndarray) -> np.ndarray:
            arr = np.unique(arr[~np.isnan(arr)])
            if arr.any():
                arr = np.sort(arr)
                max_len = len(arr)
                return np.array(
                    [arr[-min(_n, max_len)] for _n in n]
                )
            return np.array([np.nan] * len(n))

        tops = np.apply_along_axis(_top, 1, self.values.to_numpy())
        tops_labels = [f"top_{_n}" for _n in reversed(n)]

        return array_to_alike_df_or_series(
            tops,
            self.values.iloc[:, :len(n)]
        ).rename(columns=lambda _: tops_labels.pop())

    def bottom(self, n: Sequence[int]) -> pd.DataFrame:
        def _bottom(arr: np.ndarray) -> np.ndarray:
            arr = np.unique(arr[~np.isnan(arr)])
            if arr.any():
                arr = np.sort(arr)
                max_len = len(arr)
                return np.array(
                    [arr[min(_n, max_len) - 1] for _n in n]
                )
            return np.array([np.nan] * len(n))

        bottoms = np.apply_along_axis(_bottom, 1, self.values.to_numpy())
        bottoms_labels = [f"bottom_{_n}" for _n in reversed(n)]

        return array_to_alike_df_or_series(
            bottoms,
            self.values.iloc[:, :len(n)]
        ).rename(columns=lambda _: bottoms_labels.pop())
