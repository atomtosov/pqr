from __future__ import annotations

__all__ = [
    "replace_with_nan",
    "align",
    "compose",
    "partial",
    "adjust",
    "estimate_window",
    "estimate_annualizer",
    "estimate_holding",
]

from functools import (
    partial,
    reduce,
)
from typing import (
    Any,
    Callable,
    Union,
    Tuple,
)
from warnings import warn

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


def replace_with_nan(
        *values: Union[pd.DataFrame, pd.Series],
        to_replace: Any = 0
) -> Tuple[Union[pd.DataFrame, pd.Series], ...]:
    return tuple(v.replace(to_replace, np.nan) for v in values)


def align(*values: Union[pd.DataFrame, pd.Series]) -> Tuple[Union[pd.DataFrame, pd.Series], ...]:
    values = list(values)

    for i in range(len(values) - 1):
        values[i], values[i + 1] = _align_two(values[i], values[i + 1])

    for i in range(len(values) - 2, 0, -1):
        values[i], values[i - 1] = _align_two(values[i], values[i - 1])

    return tuple(values)


def _align_two(
        left: Union[pd.DataFrame, pd.Series],
        right: Union[pd.DataFrame, pd.Series]
) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
    if _are_aligned(left, right):
        return left.copy(), right.copy()

    if isinstance(left, pd.Series) or isinstance(right, pd.Series):
        return left.align(right, join="inner", axis=0)

    return left.align(right, join="inner")


def _are_aligned(
        left: Union[pd.DataFrame, pd.Series],
        right: Union[pd.DataFrame, pd.Series],
) -> bool:
    indices_aligned = ((left.index.shape == right.index.shape) and
                       (left.index == right.index).all())

    if isinstance(left, pd.Series) or isinstance(right, pd.Series):
        return indices_aligned

    columns_aligned = ((left.columns.shape == right.columns.shape) and
                       (left.columns == right.columns).all())

    return indices_aligned and columns_aligned


def compose(*funcs: Callable) -> Any:
    return reduce(
        lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)),
        funcs
    )


def adjust(
        returns: pd.Series,
        rf: Union[float, pd.Series],
) -> pd.Series:
    if isinstance(rf, pd.Series):
        returns, rf = align(returns, rf)

    return returns - rf


def estimate_window(values: Union[pd.DataFrame, pd.Series]) -> int:
    return int(estimate_annualizer(values))


def estimate_annualizer(values: Union[pd.DataFrame, pd.Series]) -> float:
    try:
        return {
            "yearly": 1.0,
            "quarterly": 4.0,
            "monthly": 12.0,
            "weekly": 52.0,
            "daily": 252.0,
        }[_estimate_periodicity(values)]
    except KeyError:
        warn("periodicity of data is undefined, estimation for annualizer is used")
        idx = values.index
        years_approx = (idx[-1] - idx[0]).days / 365.25
        return len(idx) / years_approx


def _estimate_periodicity(values: Union[pd.DataFrame, pd.Series]) -> str:
    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError(
            "values must have pd.DateTimeIndex to infer periodicity"
        )

    freqstr = pd.infer_freq(values.index)
    freq = to_offset(freqstr)

    if _is_yearly(freq):
        return "yearly"
    elif _is_quarterly(freq):
        return "quarterly"
    elif _is_monthly(freq):
        return "monthly"
    elif _is_weekly(freq):
        return "weekly"
    elif _is_daily(freq):
        return "daily"


def _is_yearly(freq: offsets.BaseOffset) -> bool:
    return isinstance(
        freq,
        (
            offsets.YearBegin,
            offsets.YearEnd,
            offsets.BYearBegin,
            offsets.BYearEnd,
            offsets.FY5253,
        )
    )


def _is_quarterly(freq: offsets.BaseOffset) -> bool:
    return isinstance(
        freq,
        (
            offsets.QuarterBegin,
            offsets.QuarterEnd,
            offsets.BQuarterBegin,
            offsets.BQuarterEnd,
            offsets.FY5253Quarter,
        )
    )


def _is_monthly(freq: offsets.BaseOffset) -> bool:
    return isinstance(
        freq,
        (
            offsets.BusinessMonthBegin,
            offsets.BusinessMonthEnd,
            offsets.CustomBusinessMonthBegin,
            offsets.CustomBusinessMonthEnd,
            offsets.MonthBegin,
            offsets.MonthEnd,
            offsets.LastWeekOfMonth,
            offsets.WeekOfMonth,
        )
    )


def _is_weekly(freq: offsets.BaseOffset) -> bool:
    return isinstance(
        freq,
        (
            offsets.Week,
        )
    )


def _is_daily(freq: offsets.BaseOffset) -> bool:
    return isinstance(
        freq,
        (
            offsets.Day,
            offsets.BusinessDay,
            offsets.CustomBusinessDay,
        )
    )


def estimate_holding(picks: pd.DataFrame) -> int:
    diff = np.diff(picks.to_numpy(), axis=0)
    rebalancings_long = (diff == 1).any(axis=1).sum()
    rebalancings_short = (diff == -1).any(axis=1).sum()
    avg_rebalacings = (rebalancings_long + rebalancings_short) / 2

    return round(len(diff) / avg_rebalacings)
