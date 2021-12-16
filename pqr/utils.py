from __future__ import annotations

from typing import Any
from warnings import warn

import numpy as np
import pandas as pd


def replace_with_nan(
        *df_or_series: pd.DataFrame | pd.Series,
        to_replace: Any = 0
) -> list[pd.DataFrame | pd.Series]:
    return [
        data.replace(to_replace, np.nan) for data in df_or_series
    ]


def is_aligned(
        left: pd.DataFrame | pd.Series,
        right: pd.DataFrame | pd.Series
) -> bool:
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
) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
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


def extract_annualizer(df_or_series: pd.DataFrame | pd.Series) -> float:
    freq_alias = {
        "A": 1, "AS": 1, "BYS": 1, "BA": 1, "BAS": 1, "RE": 1,  # yearly
        "Q": 4, "QS": 4, "BQ": 4, "BQS": 4,  # quarterly
        "M": 12, "MS": 12, "BM": 12, "BMS": 12, "CBM": 12, "CBMS": 12,  # monthly
        "W": 52,  # weekly
        "B": 252, "C": 252, "D": 252,  # daily
    }

    if not isinstance(df_or_series.index, pd.DatetimeIndex):
        raise TypeError("df or series must have pd.DateTimeIndex to infer periodicity")

    idx = df_or_series.index
    inferred_freq = getattr(idx, "inferred_freq", None)
    annualizer = freq_alias.get(inferred_freq)

    if annualizer is None:
        warn("periodicity of df or series cannot be determined correctly, estimation is used")
        years_approx = (idx[-1] - idx[0]).days / 365.25
        annualizer = len(idx) / years_approx

    return annualizer


def adjust(
        returns: pd.Series,
        rf: float | pd.Series
) -> pd.Series:
    if isinstance(rf, pd.Series):
        returns, rf = align(returns, rf)

    return returns - rf
