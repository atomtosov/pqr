import numpy as np

from .singlefactor import SingleFactor
from pqr.utils import epsilon, Interval, Quantiles, Thresholds


class Factor(SingleFactor):
    def choose(self,
               data: np.ndarray,
               interval: Interval,
               looking_period: int = 1,
               lag_period: int = 0) -> np.ndarray:
        """
        Принимает на вход данные, обрабатывает их и возвращает массив из
        True и False - выбранные по фактору позиции
        :param data:
        :param interval:
        :param looking_period:
        :param lag_period:
        :return:
        """
        # exclude values which are not available in data
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
