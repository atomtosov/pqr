"""
This module contains stuff to work with factors data. Factors are drivers of
returns on the stock (and any other) market, explaining the risks of investing
into assets. We assume that a factor can be presented by simple table (pandas
DataFrame), where each row represents a timestamp, each column - a stock
(ticker) and each cell - value of the factor for the stock in the timestamp.

You must be accurate to work with such type of data, because it is very easy to
forget about look-ahead bias and to build very profitable, but unrealistic
factor model or portfolio. There are already included "batteries" to transform
factors with looking, lag and holding periods to avoid it.

Factors can be dynamic or static. Static factor is a factor, for which is not
necessary to calculate the change, its value can be compared across the stocks
straightly (e.g. P/E). Dynamic factor is a factor, which values must be
recalculated to get the percentage changes to work with them (e.g. Momentum).
Start of testing for dynamic factors will be 1 period later to avoid look-ahead
bias.

Factors also have one more characteristic - whether values of factor is better
to be bigger or lower. For example, we usually want to pick into portfolio
stocks with low P/E, hence, this factor is "lower better", whereas stocks with
high ROA are usually more preferable, hence, this factor is "bigger better".
This characteristic affects building wml-portfolios for factor model and
weighting and scaling (leveraging) of positions in a portfolio.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    'Factor',
]


class Factor:
    """
    Class for factors, represented by matrix of numeric values.

    Parameters
    ----------
    data
        Matrix of factor values.
    """

    data: pd.DataFrame
    """Factor values."""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def transform(
            self,
            looking_back_period: int = 1,
            method: Literal['static', 'dynamic'] = 'static',
            lag_period: int = 0,
            holding_period: int = 1
    ) -> Factor:
        """
        Transforms `factor` into appropriate for decision-making format.

        This function helps to preprocess `factor` before using it to make a
        portfolio and prevents from look-ahead bias.

        Parameters
        ----------
        looking_back_period
            Looking back period for `factor`.
        method
            Whether absolute values of `factor` are used to make decision or
            their percentage changes.
        lag_period
            Delaying period to react on `factor`.
        holding_period
            Number of periods to hold each `factor` value.
        """

        return (self.look_back(looking_back_period, method)
                .lag(lag_period)
                .hold(holding_period))

    def look_back(
            self,
            period: int = 1,
            method: Literal['static', 'dynamic'] = 'static',
    ) -> Factor:
        """
        Looks back on `factor` for `period` periods.

        If `method` is dynamic:
            calculates percentage changes with looking back by `period`
            periods, then values are lagged for 1 period (because in period
            t(0) we can know percentage change from period t(-looking_period)
            only at the end of t(0), so it is needed to avoid look-ahead bias).

        If `method` is static:
            all values are shifted for `period`.

        Parameters
        ----------
        period
            Looking back period for `factor`.
        method
            Whether absolute values of `factor` are used to make decision or
            their percentage changes.
        """

        if method == 'dynamic':
            self.data = self.data.pct_change(period)
            self.lag(1)
        else:  # method = 'static'
            self.data = self.data.shift(period)

        self.data = self.data.iloc[period:]

        return self

    def lag(self, period: int = 0) -> Factor:
        """
        Simply shifts the `factor` for `period` periods.

        Can be used to simulate delayed reaction on `factor` values.

        Parameters
        ----------
        period
            Delaying period to react on `factor`.
        """

        self.data = self.data.shift(period).iloc[period:]

        return self

    def hold(self, period: int = 1) -> Factor:
        """
        Fills forward row-wise `factor` with the periodicity of `period`.

        Can be used to simulate periodical updates of `factor`.

        Parameters
        ----------
        period
            Number of periods to hold each `factor` value.
        """

        if period == 1:
            return self

        all_periods = np.zeros(len(self.data), dtype=int)
        update_periods = np.arange(len(self.data), step=period)
        all_periods[update_periods] = update_periods
        update_mask = np.maximum.accumulate(all_periods[:, np.newaxis], axis=0)

        self.data = pd.DataFrame(
            np.take_along_axis(self.data.values, update_mask, axis=0),
            index=self.data.index, columns=self.data.columns)

        return self

    def prefilter(
            self,
            mask: pd.DataFrame
    ) -> Factor:
        """
        Filters the `factor` by given `mask`.

        Simply deletes (replaces with np.nan) cells, where the `mask` equals
        to False.

        Parameters
        ----------
        mask
            Matrix of True/False, where True means that a value should remain
            in `factor` and False - that a value should be deleted.
        """

        self.data[~mask] = np.nan

        return self
