from __future__ import annotations

import functools as ft
from dataclasses import dataclass, field, InitVar
from typing import Literal, Sequence, Optional, Protocol, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from pqr.utils import align

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
    "ReplaceWithNan",
]


class Preprocessor(Protocol):
    def preprocess(self, values: pd.DataFrame) -> pd.DataFrame:
        pass


@dataclass
class Factor:
    values: pd.DataFrame = field(repr=False)
    better: Literal["more", "less"]
    preprocessor: InitVar[Optional[Preprocessor | Sequence[Preprocessor]]] = None

    def __post_init__(self, preprocessor: Optional[Preprocessor | Sequence[Preprocessor]]):
        self.values = self.values.astype(float)

        if preprocessor is not None:
            if not isinstance(preprocessor, Sequence):
                preprocessor = [preprocessor]
            for processor in preprocessor:
                self.values = processor.preprocess(self.values)

    def is_better_more(self) -> bool:
        return self.better == "more"

    def is_better_less(self) -> bool:
        return self.better == "less"

    def quantile(self, q: npt.ArrayLike[float]) -> pd.DataFrame:
        q = np.array(q).reshape((-1,))

        quantiles = np.nanquantile(
            self.values.to_numpy(),
            q if self.is_better_less() else 1 - q,
            axis=1).T

        return pd.DataFrame(
            quantiles,
            index=self.values.index,
            columns=[f"q_{_q:.2f}" for _q in q])

    def top(self, n: npt.ArrayLike[int]) -> pd.DataFrame:
        n = np.array(n).reshape((-1,))

        top_func = ft.partial(
            self._top_single if self.is_better_more() else self._bottom_single,
            n=n)

        return pd.DataFrame(
            np.apply_along_axis(top_func, 1, self.values.to_numpy()),
            index=self.values.index,
            columns=[f"top_{_n}" for _n in n])

    def bottom(self, n: npt.ArrayLike[int]) -> pd.DataFrame:
        n = np.array(n).reshape((-1,))

        bottom_func = ft.partial(
            self._bottom_single if self.is_better_more() else self._top_single,
            n=n)

        return pd.DataFrame(
            np.apply_along_axis(bottom_func, 1, self.values.to_numpy()),
            index=self.values.index,
            columns=[f"bottom_{_n}" for _n in n])

    @staticmethod
    def _top_single(arr: np.ndarray, n: np.ndarray) -> np.ndarray:
        arr = np.unique(arr[~np.isnan(arr)])
        if arr.any():
            arr = np.sort(arr)
            max_len = len(arr)
            return np.array([arr[-min(_n, max_len)] for _n in n])

        return np.array([np.nan] * len(n))

    @staticmethod
    def _bottom_single(arr: np.ndarray, n: np.ndarray) -> np.ndarray:
        arr = np.unique(arr[~np.isnan(arr)])
        if arr.any():
            arr = np.sort(arr)
            max_len = len(arr)
            return np.array([arr[min(_n, max_len) - 1] for _n in n])

        return np.array([np.nan] * len(n))


@dataclass
class Filter:
    mask: pd.DataFrame = field(repr=False)

    def __post_init__(self):
        self.mask = self.mask.astype(bool)

    def preprocess(self, values: pd.DataFrame) -> pd.DataFrame:
        universe, values = align(self.mask, values)
        return pd.DataFrame(
            np.where(universe.to_numpy(), values.to_numpy(), np.nan),
            index=values.index.copy(),
            columns=values.columns.copy())


@dataclass
class LookBackPctChange:
    period: int

    def preprocess(self, values: pd.DataFrame) -> pd.DataFrame:
        return values.pct_change(self.period).iloc[self.period:]


@dataclass
class LookBackMean:
    period: int

    def preprocess(self, values: pd.DataFrame) -> pd.DataFrame:
        return values.rolling(self.period + 1, axis=0).mean().iloc[self.period:]


@dataclass
class LookBackMedian:
    period: int

    def preprocess(self, values: pd.DataFrame) -> pd.DataFrame:
        return values.rolling(self.period + 1, axis=0).median().iloc[self.period:]


@dataclass
class LookBackMin:
    period: int

    def preprocess(self, values: pd.DataFrame) -> pd.DataFrame:
        return values.rolling(self.period + 1, axis=0).min().iloc[self.period:]


@dataclass
class LookBackMax:
    period: int

    def preprocess(self, values: pd.DataFrame) -> pd.DataFrame:
        return values.rolling(self.period + 1, axis=0).max().iloc[self.period:]


@dataclass
class Lag:
    period: int

    def preprocess(self, values: pd.DataFrame) -> pd.DataFrame:
        values = values.shift(self.period)

        if self.period >= 0:
            return values.iloc[self.period:]
        else:
            return values.iloc[:self.period]


@dataclass
class Hold:
    period: int

    def preprocess(self, values: pd.DataFrame) -> pd.DataFrame:
        periods = np.zeros(len(values), dtype=int)
        update_periods = np.arange(len(values), step=self.period)
        periods[update_periods] = update_periods
        update_mask = np.maximum.accumulate(periods[:, np.newaxis], axis=0)

        return pd.DataFrame(
            np.take_along_axis(values.to_numpy(), update_mask, axis=0),
            index=values.index.copy(),
            columns=values.columns.copy()
        )


@dataclass
class ReplaceWithNan:
    to_replace: Any

    def preprocess(self, values: pd.DataFrame) -> pd.DataFrame:
        return values.replace(self.to_replace, np.nan)
