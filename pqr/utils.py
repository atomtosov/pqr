from __future__ import annotations

import pandas as pd


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
