from __future__ import annotations

import functools as ft
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "Factor",
]


@dataclass(eq=False, frozen=True)
class Factor:
    values: pd.DataFrame = field(repr=False)
    better: Literal["more", "less"]

    # TODO: add post init check of input

    def is_better_more(self) -> bool:
        return self.better == "more"

    def is_better_less(self) -> bool:
        return self.better == "less"

    def quantile(self, q: float | Sequence[float]) -> pd.DataFrame:
        if isinstance(q, float):
            q = [q]

        quantiles = np.nanquantile(
            self.values.to_numpy(), q, axis=1
        ).T

        if self.better == "more":
            quantiles = np.fliplr(quantiles)

        return pd.DataFrame(
            quantiles,
            index=self.values.index,
            columns=[f"q_{_q:.2f}" for _q in q]
        )

    def top(self, n: int | Sequence[int]) -> pd.DataFrame:
        if isinstance(n, int):
            n = [n]

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

    def bottom(self, n: int | Sequence[int]) -> pd.DataFrame:
        if isinstance(n, int):
            n = [n]

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
    def _top_single(arr: np.ndarray, n: Sequence[int]) -> np.ndarray:
        arr = np.unique(arr[~np.isnan(arr)])
        if arr.any():
            arr = np.sort(arr)
            max_len = len(arr)
            return np.array(
                [arr[-min(_n, max_len)] for _n in n]
            )
        return np.array([np.nan] * len(n))

    @staticmethod
    def _bottom_single(arr: np.ndarray, n: Sequence[int]) -> np.ndarray:
        arr = np.unique(arr[~np.isnan(arr)])
        if arr.any():
            arr = np.sort(arr)
            max_len = len(arr)
            return np.array(
                [arr[min(_n, max_len) - 1] for _n in n]
            )
        return np.array([np.nan] * len(n))
