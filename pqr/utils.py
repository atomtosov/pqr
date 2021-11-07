from __future__ import annotations

from typing import Callable, Optional
from warnings import warn

import numpy as np
import numpy.typing as npt
import pandas as pd

FREQ_ALIAS = {
    "A": 1, "AS": 1, "BYS": 1, "BA": 1, "BAS": 1, "RE": 1,  # yearly
    "Q": 4, "QS": 4, "BQ": 4, "BQS": 4,  # quarterly
    "M": 12, "MS": 12, "BM": 12, "BMS": 12, "CBM": 12, "CBMS": 12,  # monthly
    "W": 52,  # weekly
    "B": 252, "C": 252, "D": 252,  # daily
}


def extract_annualizer(df_or_series: pd.DataFrame | pd.Series) -> float:
    if not isinstance(df_or_series.index, pd.DatetimeIndex):
        raise TypeError("df or series must have pd.DateTimeIndex to infer periodicity")

    idx = df_or_series.index
    inferred_freq = getattr(idx, "inferred_freq", None)
    annualizer = FREQ_ALIAS.get(inferred_freq)

    if annualizer is None:
        warn("periodicity of df or series cannot be determined correctly, estimation is used")
        years_approx = (idx[-1] - idx[0]).days / 365.25
        annualizer = len(idx) / years_approx

    return annualizer


def is_aligned(
        left: pd.DataFrame | pd.Series,
        right: pd.DataFrame | pd.Series
) -> tuple[pd.DataFrame | pd.Series]:
    if isinstance(left, pd.Series) or isinstance(right, pd.Series):
        return (
                (left.index.shape == right.index.shape) and (left.index == right.index).all()
        )

    return (
            (left.index.shape == right.index.shape) and (left.index == right.index).all()
            and
            (left.columns.shape == right.columns.shape) and (left.columns == right.columns).all()
    )


def align(
        left: pd.DataFrame | pd.Series,
        right: pd.DataFrame | pd.Series
) -> tuple[pd.DataFrame | pd.Series, ...]:
    if is_aligned(left, right):
        return left, right

    axis = None
    if isinstance(left, pd.Series) or isinstance(right, pd.Series):
        axis = 0
    return left.align(right, join="inner", axis=axis)


def align_many(*df_or_series: pd.DataFrame | pd.Series) -> tuple[pd.DataFrame | pd.Series, ...]:
    df_or_series = list(df_or_series)

    for i in range(len(df_or_series) - 1):
        df_or_series[i], df_or_series[i + 1] = align(df_or_series[i], df_or_series[i + 1])

    for i in range(len(df_or_series) - 2, 0, -1):
        df_or_series[i], df_or_series[i - 1] = align(df_or_series[i], df_or_series[i - 1])

    return tuple(df_or_series)


def array_to_alike_df_or_series(
        arr: np.ndarray,
        df_or_series: pd.DataFrame | pd.Series
) -> pd.DataFrame | pd.Series:
    if arr.ndim == 2:
        return pd.DataFrame(arr, index=df_or_series.index, columns=df_or_series.columns)
    else:
        return pd.Series(arr, index=df_or_series.index)


def trail(
        *df_or_series: pd.DataFrame | pd.Series,
        func: Callable[[np.ndarray, ...], npt.ArrayLike],
        window: Optional[int] = None,
        **kwargs
) -> pd.DataFrame | pd.Series:
    if window is None:
        window = round(extract_annualizer(df_or_series[0]))

    df_or_series = align_many(*df_or_series)
    arrays = [data.to_numpy() for data in df_or_series]

    values = []
    for i in range(window, len(df_or_series[0]) + 1):
        idx = slice(i - window, i)
        values.append(
            func(
                *[data[idx] for data in arrays],
                **kwargs
            )
        )

    return array_to_alike_df_or_series(
        np.array(values),
        df_or_series[0].iloc[window - 1:]
    )
