from typing import Optional, Union

import numpy as np
import pandas as pd

import pqr.thresholds

__all__ = [
    'Factor',
    'pick',
    'filter',
    'weigh',
    'scale',
]


class Factor:
    """
    Class for factors, represented by matrix of values.

    Parameters
    ----------
    data : pd.DataFrame
        Data, representing factor values.
    dynamic : bool
        Whether absolute values of factor are used to make decision or
        relative (pct changes).
    bigger_better : bool
        Whether bigger values of factor should be considered as better or on
        the contrary as worse.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 dynamic: bool = False,
                 bigger_better: bool = True):
        self.data = data
        self.dynamic = dynamic
        self.bigger_better = bigger_better

    def __repr__(self):
        return f'{type(self).__name__}()'

    def transformed(self,
                    looking_period: int,
                    lag_period: int,
                    holding_period: int) -> pd.DataFrame:
        """
        Transform factor values into appropriate for decision-making format.

        If factor is dynamic:
            calculate percentage changes with looking back for "looking_period"
            periods, then values are lagged for 1 period (because in period
            t(0) we can know percentage change from period t(-looking_period)
            only at the end of t(0), so it is needed to avoid looking-forward
            bias); then values are lagged for lag_period.

        If factor is static:
            all values are lagged for the sum of "looking_period" and
            "lag_period".

        Then factor values are forward filled with the same values, repeated
        "holding_period" times. After that operation matrix of factor values
        has each "holding_period" rows, filled with the same values.

        Parameters
        ----------
        looking_period : int, default=1
            Looking back period.
        lag_period : int, default=0
            Delaying period to react on factor.
        holding_period : int, default=1
            Number of periods to hold each factor value.

        Returns
        -------
        pd.DataFrame
            DataFrame with the same shape as given data, but with first rows,
            filled with nans. The amount of "blank" rows depends on the sum of
            "looking_period", "lag_period" and indicator that factor is
            dynamic.
        """

        if self.dynamic:
            factor = self.data.pct_change(looking_period).shift(1 + lag_period)
        else:
            factor = self.data.shift(looking_period + lag_period)

        # to save memory and speed
        if holding_period == 1:
            return factor

        start = looking_period + lag_period + self.dynamic
        all_periods = np.zeros(len(factor), dtype=int)
        rebalancing_periods = np.array(
            list(range(start, len(factor), holding_period)))
        all_periods[rebalancing_periods] = rebalancing_periods
        rebalancing_mask = np.maximum.accumulate(all_periods[:, np.newaxis],
                                                 axis=0)
        held_factor = np.take_along_axis(factor.values, rebalancing_mask,
                                         axis=0)

        return pd.DataFrame(held_factor,
                            index=factor.index,
                            columns=factor.columns)


def pick(data: pd.DataFrame,
                   factor: Optional[Factor] = None,
                   thresholds: Optional[pqr.thresholds.Thresholds] =
                   pqr.thresholds.Thresholds(),
                   looking_period: int = 1,
                   lag_period: int = 0,
                   holding_period: int = 1) -> pd.DataFrame:
    """
    Pick stocks by given interval of factor values with respect to given data.

    Parameters
    ----------
    data : pd.DataFrame
        Data, representing stock universe (implied that this data is
        prices). If some values are missed in data but exist in factor
        values, they are excluded from factor values too to prevent
        situations, when stock cannot be actually traded, but picked.
    factor : Factor, optional
        Factor used to pick values from data universe. If not given just pick
        all non-nan values from data.
    thresholds : Thresholds, default=Thresholds()
        Represents interval of factor values to pick.
    looking_period : int, default=1
        Looking back period.
    lag_period : int, default=0
        Delaying period to react on factor.
    holding_period : int, default=1
        Number of periods to hold each pick.

    Returns
    -------
    pd.DataFrame
        DataFrame of True/False, where True shows that a stock is fall into
        given interval and should be picked into portfolio, and False shows
        that a stock shouldn't.

    Notes
    -----
    Supports picking values from data universe by quantiles and tops, other
    types of intervals are interpreted as simple constant thresholds.
    """

    if factor is None:
        return pd.DataFrame(~np.isnan(data.values), index=data.index,
                            columns=data.columns)

    factor_values = factor.transformed(looking_period, lag_period,
                                       holding_period)
    factor_values.values[np.isnan(data.values)] = np.nan

    if isinstance(thresholds, pqr.thresholds.Quantiles):
        lower_threshold = np.nanquantile(factor_values.values,
                                         thresholds.lower,
                                         axis=1)[:, np.newaxis]
        upper_threshold = np.nanquantile(factor_values.values,
                                         thresholds.upper,
                                         axis=1)[:, np.newaxis]
    elif isinstance(thresholds, pqr.thresholds.Top):
        lower_threshold = np.nanmin(factor_values.apply(pd.Series.nlargest,
                                                        n=thresholds.lower,
                                                        axis=1),
                                    axis=1)[:, np.newaxis]
        upper_threshold = np.nanmin(factor_values.apply(pd.Series.nlargest,
                                                        n=thresholds.upper,
                                                        axis=1),
                                    axis=1)[:, np.newaxis]
    else:
        lower_threshold = thresholds.lower
        upper_threshold = thresholds.upper

    return pd.DataFrame((lower_threshold <= factor_values) &
                        (factor_values <= upper_threshold),
                        index=data.index, columns=data.columns, dtype=bool)


def filter(data: pd.DataFrame,
                     factor: Optional[Factor] = None,
                     thresholds: pqr.thresholds.Thresholds =
                     pqr.thresholds.Thresholds(),
                     looking_period: int = 1,
                     lag_period: int = 0,
                     holding_period: int = 1) -> pd.DataFrame:
    """
    Filter data universe by given interval of factor values with respect to
    given data.

    Parameters
    ----------
    data : pd.DataFrame
        Data to be filtered by factor values. Expected to get stock prices,
        but it isn't obligatory.
    factor : Factor, optional
        Factor used to exclude values from data universe. If not given just
        don't filter stock universe at all.
    thresholds : Thresholds, default=Thresholds()
        Represents interval of factor values to be included. All points, where
        factor values are not in range, are excluded.
    looking_period : int, default=1
        Looking back period.
    lag_period : int, default=0
        Delaying period to react on factor.
    holding_period : int, default=1
        Number of periods to hold filter.

    Returns
    -------
    pd.DataFrame
        Dataframe with the same data as given, but filled with nans in
        filtered places.
    """

    in_range = pick(data, factor, thresholds, looking_period, lag_period,
                    holding_period)
    filtered_values = data.values.copy().astype(float)
    filtered_values[~in_range.values] = np.nan
    return pd.DataFrame(filtered_values,
                        index=data.index,
                        columns=data.columns)


def weigh(data: pd.DataFrame,
                    factor: Optional[Factor] = None,
                    looking_period: int = 1,
                    lag_period: int = 0,
                    holding_period: int = 1) -> pd.DataFrame:
    """
    Weigh values in given data by factor values.

    Finds linear weights: simply divides each value in a row by the sum of
    the row. If factor is lower_better (bigger_better=False), then weights are
    additionally "mirrored".

    Parameters
    ----------
    data : pd.DataFrame
        Data to be weighted. It is implied to get positions (matrix with
        True/False), but it is not obligatory: if data doesn't represent
        positions weights are affected by values of given data.
    factor : Factor, optional
        Factor used to weigh values in data. If not given weigh with simple
        equal weights.
    looking_period : int, default=1
        Looking back period.
    lag_period : int, default=0
        Delaying period to react on factor.
    holding_period : int, default=1
        Number of periods to hold weights similar.

    Returns
    -------
    pd.DataFrame
        DataFrame with weights for given data. It is guaranteed that the
        sum of values in each row is equal to 1.

    Notes
    -----
    Works only for factors with all positive values.
    """

    if factor is None:
        factor = Factor(pd.DataFrame(np.ones_like(data.values, dtype=int),
                                     index=data.index, columns=data.columns))

    factor_values = factor.transformed(looking_period, lag_period,
                                       holding_period).values
    values = data.values * factor_values
    weights = values / np.nansum(values, axis=1)[:, np.newaxis]

    if not factor.bigger_better:
        straight_sort = np.argsort(weights, axis=1)
        reversed_sort = np.fliplr(straight_sort)
        for i in range(len(weights)):
            weights[i, straight_sort[i]] = weights[i, reversed_sort[i]]

    return pd.DataFrame(weights, index=data.index,
                        columns=data.columns).fillna(0)


def scale(data: pd.DataFrame,
                    factor: Optional[Factor] = None,
                    target: Union[int, float] = 1,
                    looking_period: int = 1,
                    lag_period: int = 0,
                    holding_period: int = 1) -> pd.DataFrame:
    """
    Scaling values in given data by factor values and target.

    Simply divides each value in a row by target. If factor is lower_better
    (bigger_better=False), then leverage additionally "mirrored".

    Parameters
    ----------
    data : pd.DataFrame
        Data to be leveraged. It is implied to get weights (matrix with
        each row sum equals to 1), but it is not obligatory.
    factor : Factor, optional
        Factor used to leverage values in data. If not given just returns the
        same data (like it was scaled by leverage 1 every time).
    target : int, float, default=1
        Target to scale positions. Used as divisor to get scaling values.
    looking_period : int, default=1
        Looking back period.
    lag_period : int, default=0
        Delaying period to react on factor.
    holding_period : int, default=1
        Number of periods to hold leverage similar.

    Returns
    -------
    pd.DataFrame
        Dataframe with scaled by leverage values.

    Notes
    -----
    Works only for factors with all positive values.
    """

    if factor is None:
        return data

    factor_values = factor.transformed(looking_period, lag_period,
                                       holding_period).values

    values = data.values * factor_values
    leverage = values / target
    if not factor.bigger_better:
        straight_sort = np.argsort(leverage, axis=1)
        reversed_sort = np.fliplr(straight_sort)
        for i in range(len(leverage)):
            leverage[i, straight_sort[i]] = leverage[i, reversed_sort[i]]

    return pd.DataFrame(leverage, index=data.index,
                        columns=data.columns).fillna(0)
