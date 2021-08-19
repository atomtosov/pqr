"""
This module contains stuff to work with factors data. Factors are drivers of returns on the stock
(and any other) market, explaining the risks of investing into assets. We assume that a factor can
be presented by simple table (pandas DataFrame), where each row represents a timestamp, each
column - a stock (ticker) and each cell - value of the factor for the stock in the timestamp.

You must be accurate to work with such type of data, because it is very easy to forget about
look-ahead bias and to build very profitable, but unrealistic factor model or portfolio. There are
already included "batteries" to transform factors with looking, lag and holding periods to avoid it.

Factors can be dynamic or static. Static factor is a factor, for which is not necessary to calculate
the change, its value can be compared across the stocks straightly (e.g. P/E). Dynamic factor is
a factor, which values must be recalculated to get the percentage changes to work with them
(e.g. Momentum). Start of testing for dynamic factors will be 1 period later to avoid look-ahead
bias.

Factors also have one more characteristic - whether values of factor is better to be bigger or
lower. For example, we usually want to pick into portfolio stocks with low P/E, hence, this factor
is "lower better", whereas stocks with high ROA are usually more preferable, hence, this factor is
"bigger better". This characteristic affects building wml-portfolios.
"""

import numpy as np
import pandas as pd

from .utils import align

__all__ = [
    'Factor',
]


class Factor:
    """
    Class for factors, represented by matrix of numeric values.

    Parameters
    ----------
    data : pd.DataFrame
        Matrix of factor values.
    """

    data: pd.DataFrame
    """Factor values."""

    def __init__(self, data):
        self.data = data.copy()

    def look_back(self, period=1, method='static'):
        """
        Looks back on `factor` for `period` periods.

        If `method` is dynamic:
            calculates percentage changes with looking back by `period` periods, then values are
            lagged for 1 period (because in period t(0) we can know percentage change from period
            t(-looking_period) only at the end of t(0), so it is needed to avoid look-ahead bias).

        If `method` is static:
            all values are shifted for `period`.

        Parameters
        ----------
        period : int > 0, default=1
            Looking back period for `factor`.
        method : {'static', 'dynamic'}, default='static'
            Whether absolute values of `factor` are used to make decision or their percentage
            changes.

        Returns
        -------
        Factor
            Factor with transformed data.
        """

        if method == 'dynamic':
            self.data = self.data.pct_change(period)
            self.lag(1)
        else:  # method = 'static'
            self.data = self.data.shift(period)

        self.data = self.data.iloc[period:]

        return self

    def lag(self, period=0):
        """
        Simply shifts the `factor` for `period` periods.

        Can be used to simulate delayed reaction on `factor` values.

        Parameters
        ----------
        period : int >= 0, default=0
            Delaying period to react on `factor`.

        Returns
        -------
        Factor
            Factor with transformed data.
        """

        self.data = self.data.shift(period).iloc[period:]

        return self

    def hold(self, period=1):
        """
        Fills forward row-wise `factor` with the periodicity of `period`.

        Can be used to simulate periodical updates of `factor`.

        Parameters
        ----------
        period : int > 0, default=1
            Number of periods to hold each `factor` value.

        Returns
        -------
        Factor
            Factor with transformed data.
        """

        # to avoid useless computations below
        if period == 1:
            return self

        periods = np.zeros(len(self.data), dtype=int)
        update_periods = np.arange(len(self.data), step=period)
        periods[update_periods] = update_periods
        update_mask = np.maximum.accumulate(periods[:, np.newaxis], axis=0)

        self.data = pd.DataFrame(
            np.take_along_axis(self.data.values, update_mask, axis=0),
            index=self.data.index, columns=self.data.columns
        )

        return self

    def prefilter(self, mask):
        """
        Filters the `factor` by given `mask`.

        Simply deletes (replaces with np.nan) cells, where the `mask` equals to False.

        Parameters
        ----------
        mask : pd.DataFrame
            Matrix of True/False, where True means that a value should remain in `factor` and
            False - that a value should be deleted.

        Returns
        -------
        Factor
            Factor with transformed data.
        """

        self.data, mask = align(self.data, mask)

        self.data[~mask] = np.nan

        return self
