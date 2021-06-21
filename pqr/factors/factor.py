import numpy as np

from .singlefactor import SingleFactor
from pqr.utils import epsilon, Interval, Quantiles, Thresholds


class Factor(SingleFactor):
    """
    Class Factor, which actually represents ChoosingFactor - Factor, which can
    choose some stocks by its own transformed values with respect to given data
    Extends SingleFactor

    Attributes:
        dynamic: bool - is factor dynamic or not, this information is needed
        for future transformation of factor data
        bigger_better: bool | None - is better, when factor value bigger
        (e.g. ROA) or when factor value lower (e.g. P/E); if value is None it
        means that it cannot be said exactly, what is better (used for multi-
        factors)
        periodicity: DataPeriodicity - info about periodicity or discreteness
        of factor data, used for annualization and smth more
        name: str - name of factor

    Methods:
        transform() - returns transformed values of factor data
        with looking_period and lag_period (NOTE: if factor is dynamic,
        real lag = lag_period + 1)

        choose() - choose values from given dataset with respect to its own
        values, transformed with looking_period and lag_period, and interval
    """
    def choose(self,
               data: np.ndarray,
               interval: Interval,
               looking_period: int = 1,
               lag_period: int = 0) -> np.ndarray:
        """

        :param data: dataset, from which pick stocks (it is needed to exclude
        from factor data values, when in the same period are not represented in
        data (e.g. after filtering))
        :param interval: interval by which pick stocks
        :param looking_period: period to lookahead
        :param lag_period: period to shift data
        :return: 2-dimensional boolean matrix of choices (True if chosen)
        """
        # exclude values which are not available in data (e.g. after filtering)
        values = self.transform(looking_period, lag_period)
        values[np.isnan(data)] = np.nan

        if isinstance(interval, Quantiles):
            lower_threshold = np.nanquantile(values, interval.lower, axis=1)
            upper_threshold = np.nanquantile(values, interval.upper, axis=1)
            # to include stock with highest factor value
            if interval.upper == 1:
                upper_threshold += epsilon
            choice = (lower_threshold[:, np.newaxis] <= values) & \
                     (values < upper_threshold[:, np.newaxis])
            data = (data * choice).astype(float)
            data[data == 0] = np.nan
            return ~np.isnan(data)
        elif isinstance(interval, Thresholds):
            choice = (interval.lower <= values) & (values < interval.upper)
            data = (data * choice).astype(float)
            data[data == 0] = np.nan
            return ~np.isnan(data)
        else:
            raise ValueError('interval must be Quantiles or Thresholds')
