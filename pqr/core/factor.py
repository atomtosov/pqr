from __future__ import annotations

import functools as ft
from typing import Callable, Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from pqr.utils import align, compose
from .universe import Universe

__all__ = [
    "Factor",

    "Factorizer",

    "Filter",
    "Lag",
    "Hold",
]


class Factor:
    def __init__(
            self,
            values: pd.DataFrame,
            better: Literal["more", "less"]
    ):
        self.values = values.astype(float)
        self.better = better

    def is_better_more(self) -> bool:
        return self.better == "more"

    def is_better_less(self) -> bool:
        return self.better == "less"

    def quantile(self, q: npt.ArrayLike[float]) -> pd.DataFrame:
        q = np.array(q).reshape((-1,))

        quantiles = np.nanquantile(
            self.values.to_numpy(),
            q if self.is_better_less() else 1 - q,
            axis=1
        ).T

        return pd.DataFrame(
            quantiles,
            index=self.values.index,
            columns=[f"q_{_q:.2f}" for _q in q]
        )

    def top(self, n: npt.ArrayLike[int]) -> pd.DataFrame:
        n = np.array(n).reshape((-1,))

        if self.better == "more":
            top_func = self._top_single
        else:  # better = "less"
            top_func = self._bottom_single
        top_func = ft.partial(top_func, n=n)

        return pd.DataFrame(
            np.apply_along_axis(top_func, 1, self.values.to_numpy()),
            index=self.values.index,
            columns=[f"top_{_n}" for _n in n]
        )

    def bottom(self, n: npt.ArrayLike[int]) -> pd.DataFrame:
        n = np.array(n).reshape((-1,))

        if self.better == "more":
            bottom_func = self._bottom_single
        else:  # better = "less"
            bottom_func = self._top_single
        bottom_func = ft.partial(bottom_func, n=n)

        return pd.DataFrame(
            np.apply_along_axis(bottom_func, 1, self.values.to_numpy()),
            index=self.values.index,
            columns=[f"bottom_{_n}" for _n in n]
        )

    @staticmethod
    def _top_single(arr: np.ndarray, n: np.ndarray) -> np.ndarray:
        arr = np.unique(arr[~np.isnan(arr)])
        if arr.any():
            arr = np.sort(arr)
            max_len = len(arr)
            return np.array(
                [arr[-min(_n, max_len)] for _n in n]
            )
        return np.array([np.nan] * len(n))

    @staticmethod
    def _bottom_single(arr: np.ndarray, n: np.ndarray) -> np.ndarray:
        arr = np.unique(arr[~np.isnan(arr)])
        if arr.any():
            arr = np.sort(arr)
            max_len = len(arr)
            return np.array(
                [arr[min(_n, max_len) - 1] for _n in n]
            )
        return np.array([np.nan] * len(n))


class Factorizer:
    def __init__(self, *preprocessors: Callable[[pd.DataFrame], pd.DataFrame]):
        self._preprocessor = compose(*preprocessors)

    def __call__(
            self,
            values: pd.DataFrame,
            better: Literal["more", "less"],
    ) -> Factor:
        return Factor(
            self._preprocessor(values),
            better
        )


class Filter:
    def __init__(self, mask: pd.DataFrame):
        self.mask = mask.astype(bool)

    def __call__(self, values: pd.DataFrame) -> pd.DataFrame:
        universe, values = align(self.mask, values)
        return pd.DataFrame(
            np.where(universe.to_numpy(), values.to_numpy(), np.nan),
            index=values.index,
            columns=values.columns
        )


class Lag:
    def __init__(self, period: int):
        self.period = period

    def __call__(self, values: pd.DataFrame) -> pd.DataFrame:
        if self.period == 0:
            return values

        return pd.DataFrame(
            values.to_numpy()[:-self.period],
            index=values.index[self.period:],
            columns=values.columns
        )


class Hold:
    def __init__(self, period: int):
        self.period = period

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
