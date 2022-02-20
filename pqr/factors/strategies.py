from __future__ import annotations

__all__ = [
    "quantiles",
    "top",
    "bottom",
    "time_series",
    "split_quantiles",
    "split_top_bottom",
    "split_time_series",
]

from typing import Callable, Literal, Generator, Sequence

import numpy as np
import pandas as pd

from pqr.utils import partial


def factor_portfolios_names_factory(n: int) -> Generator[str]:
    for i in range(n):
        if i == 0:
            yield "Winners"
        elif i == n - 1:
            yield "Losers"
        else:
            yield f"Neutral {i}"


def quantiles(
        factor: pd.DataFrame,
        min_q: float = 0.0,
        max_q: float = 1.0,
) -> pd.DataFrame:
    factor_array = factor.to_numpy()
    min_q, max_q = np.nanquantile(
        factor_array,
        [min_q, max_q],
        axis=1, keepdims=True,
    )
    return pd.DataFrame(
        (min_q <= factor_array) & (factor_array <= max_q),
        index=factor.index.copy(),
        columns=factor.columns.copy(),
    )


def split_quantiles(
        n: int,
        better: Literal["more", "less"],
) -> dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
    q = np.linspace(0, 1, n + 1)
    q_strategies = [
        partial(quantiles, min_q=q[i], max_q=q[i + 1])
        for i in range(n)
    ]
    if better == "more":
        q_strategies.reverse()

    return dict(zip(factor_portfolios_names_factory(n), q_strategies))


def top(
        factor: pd.DataFrame,
        k: int = 10,
) -> pd.DataFrame:
    factor_array = factor.to_numpy()
    top_k = np.apply_along_axis(
        partial(_top_single, k=k),
        axis=1,
        arr=factor_array
    )[:, np.newaxis]
    return pd.DataFrame(
        factor_array >= top_k,
        index=factor.index.copy(),
        columns=factor.columns.copy()
    )


def bottom(
        factor: pd.DataFrame,
        k: int = 10,
) -> pd.DataFrame:
    factor_array = factor.to_numpy()
    bottom_k = np.apply_along_axis(
        partial(_bottom_single, k=k),
        axis=1,
        arr=factor_array
    )[:, np.newaxis]
    return pd.DataFrame(
        factor_array <= bottom_k,
        index=factor.index.copy(),
        columns=factor.columns.copy()
    )


def split_top_bottom(
        k: int,
        better: Literal["more", "less"],
) -> dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
    strategies = [
        partial(top, k=k),
        partial(bottom, k=k)
    ]

    if better == "less":
        strategies.reverse()

    return dict(zip(factor_portfolios_names_factory(2), strategies))


def _top_single(
        arr: np.ndarray,
        k: int,
) -> np.ndarray:
    uniq_arr = np.unique(arr[~np.isnan(arr)])
    max_k = len(uniq_arr)

    if max_k > k:
        return np.sort(uniq_arr)[-k]
    elif max_k > 0:
        return np.max(uniq_arr)
    else:
        return np.nan


def _bottom_single(
        arr: np.ndarray,
        k: int,
) -> np.ndarray:
    uniq_arr = np.unique(arr[~np.isnan(arr)])
    max_k = len(uniq_arr)

    if max_k > k:
        return np.sort(uniq_arr)[k - 1]
    elif max_k > 0:
        return np.min(uniq_arr)
    else:
        return np.nan


def time_series(
        factor: pd.DataFrame,
        min_threshold: float = -np.inf,
        max_threshold: float = np.inf,
) -> pd.DataFrame:
    factor_array = factor.to_numpy()
    return pd.DataFrame(
        (min_threshold <= factor_array) & (factor_array <= max_threshold),
        index=factor.index.copy(),
        columns=factor.columns.copy()
    )


def split_time_series(
        thresholds: Sequence[float],
        better: Literal["more", "less"],
) -> dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
    thresholds = list(sorted(thresholds))
    thresholds.insert(0, -np.inf)
    thresholds.append(np.inf)

    strategies = [
        partial(
            time_series,
            min_threshold=thresholds[i],
            max_threshold=thresholds[i + 1]
        )
        for i in range(len(thresholds) - 1)
    ]

    if better == "more":
        strategies.reverse()

    return dict(zip(
        factor_portfolios_names_factory(len(thresholds) - 1),
        strategies,
    ))
