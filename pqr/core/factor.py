from __future__ import annotations

import functools as ft
from typing import Callable, Literal, Sequence, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from pqr.utils import align
from .utils import compose, top_single, bottom_single

__all__ = [
    "Factor",
    "Preprocessor",

    "Filter",
    "LookBackPctChange",
    "LookBackMean",
    "LookBackMedian",
    "LookBackMin",
    "LookBackMax",
    "Lag",
    "Hold",
]

Preprocessor = Callable[[pd.DataFrame], pd.DataFrame]


class Factor:
    def __init__(
            self,
            values: pd.DataFrame,
            better: Literal["more", "less"],
            preprocessor: Optional[Preprocessor | Sequence[Preprocessor]] = None,
    ):
        self.values = values.astype(float)
        if isinstance(preprocessor, Sequence):
            self.values = compose(*preprocessor)(self.values)
        elif preprocessor is not None:
            self.values = preprocessor(self.values)

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

        top_func = ft.partial(
            top_single if self.is_better_more() else bottom_single,
            n=n
        )

        return pd.DataFrame(
            np.apply_along_axis(top_func, 1, self.values.to_numpy()),
            index=self.values.index,
            columns=[f"top_{_n}" for _n in n]
        )

    def bottom(self, n: npt.ArrayLike[int]) -> pd.DataFrame:
        n = np.array(n).reshape((-1,))

        bottom_func = ft.partial(
            bottom_single if self.is_better_more() else top_single,
            n=n
        )

        return pd.DataFrame(
            np.apply_along_axis(bottom_func, 1, self.values.to_numpy()),
            index=self.values.index,
            columns=[f"bottom_{_n}" for _n in n]
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


class LookBackPctChange:
    def __init__(self, period: int):
        self.period = period

    def __call__(self, values: pd.DataFrame) -> pd.DataFrame:
        return values.pct_change(self.period).iloc[self.period:]


class LookBackMean:
    def __init__(self, period: int):
        self.period = period

    def __call__(self, values: pd.DataFrame) -> pd.DataFrame:
        return values.rolling(self.period + 1, axis=0).mean().iloc[self.period:]


class LookBackMedian:
    def __init__(self, period: int):
        self.period = period

    def __call__(self, values: pd.DataFrame) -> pd.DataFrame:
        return values.rolling(self.period + 1, axis=0).median().iloc[self.period:]


class LookBackMin:
    def __init__(self, period: int):
        self.period = period

    def __call__(self, values: pd.DataFrame) -> pd.DataFrame:
        return values.rolling(self.period + 1, axis=0).min().iloc[self.period:]


class LookBackMax:
    def __init__(self, period: int):
        self.period = period

    def __call__(self, values: pd.DataFrame) -> pd.DataFrame:
        return values.rolling(self.period + 1, axis=0).max().iloc[self.period:]


class Lag:
    def __init__(self, period: int):
        self.period = period

    def __call__(self, values: pd.DataFrame) -> pd.DataFrame:
        if self.period == 0:
            return values

        return pd.DataFrame(
            values.to_numpy()[:-self.period],
            index=values.index[self.period:].copy(),
            columns=values.columns.copy()
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
            index=values.index.copy(),
            columns=values.columns.copy()
        )
