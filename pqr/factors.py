"""
This module contains stuff to work with factors data. Factors are drivers of
returns on the stock (and any other) market, explaining the risks of investing
into assets. We assume that a factor can be presented by simple table (pandas
DataFrame), where each row represents a timestamp, each column - a stock
(ticker) and each cell - value of the factor for the stock in the timestamp.

You must be accurate to work with such type of data, because it is very easy to
forget about look-ahead bias and build very profitable, but unrealistic factor
model or portfolio. There are already included "batteries" to transform factors
with looking, lag and holding periods to avoid it.

Factors can be dynamic or static. Static factor is a factor, for which is not
necessary to calculate the change, its value can be compared across the stocks
straightly (e.g. P/E). Dynamic factor is a factor, which values must be
recalculated to get the percentage change to work with them (e.g. Momentum).
Start of testing for dynamic factors will be 1 period later to avoid look-ahead
bias.

Factors also have one more characteristic - whether values of factor is better
to be bigger or lower. For example, we usually want to pick into portfolio
stocks with low P/E, hence, this factor is "lower better", whereas stocks with
high ROA are usually more preferable, hence, this factor is "bigger better".
This characteristic affects building wml-portfolios for factor model and
weighting and scaling (leveraging) of positions in a portfolio.
"""


import functools as ft
from typing import Optional, Union, Callable

import numpy as np
import pandas as pd

import pqr.thresholds


__all__ = [
    'factorize',
    'filtrate',
    'select',
    'weigh',
    'scale',
]


def factorize(
        factor: pd.DataFrame,
        is_dynamic: bool = False,
        looking_period: int = 1,
        lag_period: int = 0,
        holding_period: int = 1
) -> pd.DataFrame:
    """
    Transforms `factor` into appropriate for decision-making view.

    If `factor` is dynamic:
        calculates percentage changes with looking back by `looking_period`
        periods, then values are lagged for 1 period (because in period
        t(0) we can know percentage change from period t(-looking_period)
        only at the end of t(0), so it is needed to avoid looking-forward
        bias); then values are lagged for `lag_period`.

    If `factor` is static:
        all values are lagged for the sum of `looking_period` and `lag_period`.

    Then factor values are forward filled with the same values, repeated
    `holding_period` times. After that operation matrix of factor values
    has each `holding_period` rows, filled with the same values.

    Parameters
    ----------
    factor
        Factor data to be transformed.
    is_dynamic
        Whether absolute values of `factor` are used to make decision or their
        percentage changes.
    looking_period
        Looking back period for `factor`.
    lag_period
        Delaying period to react on `factor`.
    holding_period
        Number of periods to hold each factor value.

    """

    if is_dynamic:
        data = factor.pct_change(looking_period).shift(1 + lag_period)
    else:
        data = factor.shift(looking_period + lag_period)
    start = looking_period + lag_period + is_dynamic
    data = data.iloc[start:]

    if holding_period > 1:
        all_periods = np.zeros(len(data), dtype=int)
        update_periods = np.arange(len(data), step=holding_period)
        all_periods[update_periods] = update_periods
        update_mask = np.maximum.accumulate(all_periods[:, np.newaxis], axis=0)
        data.values[:] = np.take_along_axis(data.values, update_mask, axis=0)

    return data


def _validate_data_and_factor_data(func: Callable) -> Callable:
    """
    Decorator, fixing the problem of different shapes of transformed factor and
    passed data.
    """

    @ft.wraps(func)
    def validated_func(data: pd.DataFrame,
                       factor: Optional[pd.DataFrame] = None,
                       *args, **kwargs):
        if factor is not None:
            min_index = max(data.index[0], factor.index[0])
            data = data.loc[min_index:]
            factor = factor.loc[min_index:]
        return func(data, factor, *args, **kwargs)

    return validated_func


@_validate_data_and_factor_data
def select(
        data: pd.DataFrame,
        factor: Optional[pd.DataFrame] = None,
        thresholds: pqr.thresholds.Thresholds = pqr.thresholds.Thresholds()
) -> pd.DataFrame:
    """
    Select stocks from the `data` by given interval of `factor`.

    If some values are missed in the data but exist in `factor` values, they
    are excluded from `factor` values too to prevent situations, when stock
    cannot be actually traded, but selected into portfolio.

    Parameters
    ----------
    data
        Data, representing stock universe (implied that this data is prices).
    factor
        Factor, used for selecting stocks from stock universe. If not passed,
        simply select all available for trading stocks.
    thresholds
        Bounds for the set of allowed values of `factor` to select stocks.

    Notes
    -----
    Supports picking values from data universe by quantiles and tops, other
    types of intervals are interpreted as simple constant thresholds.
    """

    if factor is None:
        return ~data.isna()

    factor = factor.copy().astype(float)
    factor.values[np.isnan(data.values)] = np.nan
    if isinstance(thresholds, pqr.thresholds.Quantiles):
        lower_threshold, upper_threshold = np.nanquantile(
            factor, [thresholds.lower, thresholds.upper],
            axis=1, keepdims=True
        )
    elif isinstance(thresholds, pqr.thresholds.Top):
        lower_threshold = np.nanmin(
            factor.apply(pd.Series.nlargest, n=thresholds.lower, axis=1),
            axis=1
        )[:, np.newaxis]
        upper_threshold = np.nanmin(
            factor.apply(pd.Series.nlargest, n=thresholds.upper, axis=1),
            axis=1
        )[:, np.newaxis]
    else:
        lower_threshold = thresholds.lower
        upper_threshold = thresholds.upper

    return pd.DataFrame(
        (lower_threshold <= factor) & (factor <= upper_threshold),
        index=factor.index, columns=factor.columns
    )


@_validate_data_and_factor_data
def filtrate(
        data: pd.DataFrame,
        factor: Optional[pd.DataFrame] = None,
        thresholds: pqr.thresholds.Thresholds = pqr.thresholds.Thresholds()
) -> pd.DataFrame:
    """
    Filtrate the `data` by given interval of `factor`.

    Actually, it just selects stocks from stock universe, and those are not
    selected replaced with np.nan.

    Parameters
    ----------
    data
        Data to be filtered by factor values (implied that this data is
        prices).
    factor
        Factor, used as filter. If not passed, don't filtrate at all.
    thresholds
        Bounds for the set of allowed values of `factor`.
    """

    if factor is None:
        return data

    in_range = select(data, factor, thresholds)
    data = data.copy().astype(float)
    data.values[~in_range.values] = np.nan
    return data


@_validate_data_and_factor_data
def weigh(
        data: pd.DataFrame,
        factor: Optional[pd.DataFrame] = None,
        is_bigger_better: bool = True
) -> pd.DataFrame:
    """
    Weigh the `data` by `factor` values.

    Finds linear weights: simply divides each value in a row by the sum of
    the row. If factor is lower_better (`is_bigger_better` = False), then
    weights are additionally "mirrored".

    Parameters
    ----------
    data
        Data to be weighted. It is implied to get positions (matrix with
        True/False). If data doesn't represent positions, weights are affected
        by values of given data.
    factor
        Factor, used to weigh positions. If not passed, just weighs equally.
    is_bigger_better
        Whether bigger values of `factor` will lead to bigger weights for a
        position or on the contrary to lower.

    Notes
    -----
    Works only for factors with all positive values.
    """

    if factor is None:
        factor = np.ones_like(data, dtype=int)

    data = data * factor
    weights = (data / data.sum(axis=1)[:, np.newaxis]).fillna(0)
    if not is_bigger_better:
        straight_sort = np.argsort(weights, axis=1)
        reversed_sort = np.fliplr(straight_sort)
        for i in range(len(weights)):
            weights[i, straight_sort[i]] = weights[i, reversed_sort[i]]

    return weights


@_validate_data_and_factor_data
def scale(
        data: pd.DataFrame,
        factor: Optional[pd.DataFrame] = None,
        is_bigger_better: bool = True,
        target: Union[int, float] = 1
) -> pd.DataFrame:
    """
    Scale the `data` by target of `factor`.

    Simply divides each value in a row by `target`. If `factor` is lower_better
    (`bigger_better` = False), then leverages is additionally "mirrored".

    Parameters
    ----------
    data
        Data to be leveraged. It is implied to get weights (matrix with
        each row sum equals to 1).
    factor
        Factor, used to scale (leverage) positions weights. If not passed,
        do not scale at all.
    is_bigger_better
        Whether bigger values of `factor` will lead to bigger leverage for a
        position or on the contrary to lower.
    target
        Target of `factor` values.

    Notes
    -----
    Works only for factors with all positive values.
    """

    if factor is None:
        return data

    leverage = data * factor.data / target
    if not is_bigger_better:
        straight_sort = np.argsort(leverage, axis=1)
        reversed_sort = np.fliplr(straight_sort)
        for i in range(len(leverage)):
            leverage[i, straight_sort[i]] = leverage[i, reversed_sort[i]]

    return leverage
