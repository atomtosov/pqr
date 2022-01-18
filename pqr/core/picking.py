__all__ = [
    "pick",
    "quantiles",
    "top",
    "bottom",
    "time_series",
    "split_quantiles",
    "split_top_bottom",
    "split_time_series",
]

from typing import (
    Callable,
    Optional,
    List,
)

import numpy as np
import pandas as pd

from pqr.utils import (
    align,
    partial,
)


def pick(
        longs: Optional[pd.DataFrame] = None,
        shorts: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if longs is None and shorts is None:
        raise ValueError("either longs or shorts must be given")
    elif longs is not None and shorts is not None:
        longs, shorts = align(longs, shorts)
        picks = longs.astype(np.int8) - shorts.astype(np.int8)
    elif longs is not None and shorts is None:
        picks = longs.astype(np.int8)
    else:
        picks = shorts.astype(np.int8)

    return picks


def quantiles(
        factor: pd.DataFrame,
        min_q: float = 0.0,
        max_q: float = 1.0
) -> pd.DataFrame:
    factor_array = factor.to_numpy()
    min_q, max_q = np.nanquantile(
        factor_array,
        [min_q, max_q],
        axis=1, keepdims=True
    )

    return pd.DataFrame(
        (min_q <= factor_array) & (factor_array <= max_q),
        index=factor.index.copy(),
        columns=factor.columns.copy()
    )


def split_quantiles(n: int) -> List[Callable[[pd.DataFrame], pd.DataFrame]]:
    q = np.linspace(0, 1, n + 1)
    return [
        partial(quantiles, min_q=q[i], max_q=q[i + 1])
        for i in range(n)
    ]


def top(
        factor: pd.DataFrame,
        k: int = 10
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
        k: int = 10
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


def split_top_bottom(k: int) -> List[Callable[[pd.DataFrame], pd.DataFrame]]:
    return [
        partial(top, k=k),
        partial(bottom, k=k)
    ]


def time_series(
        factor: pd.DataFrame,
        min_threshold: float = -np.inf,
        max_threshold: float = np.inf
) -> pd.DataFrame:
    factor_array = factor.to_numpy()

    return pd.DataFrame(
        (min_threshold <= factor_array) & (factor_array <= max_threshold),
        index=factor.index.copy(),
        columns=factor.columns.copy()
    )


def split_time_series(threshold: float = 0.0) -> List[Callable[[pd.DataFrame], pd.DataFrame]]:
    return [
        partial(time_series, max_threshold=threshold),
        partial(time_series, min_threshold=threshold),
    ]


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
