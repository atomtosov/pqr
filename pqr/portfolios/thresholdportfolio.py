from typing import Union

from .intervalportfolio import IntervalPortfolio
from pqr.utils import Thresholds


class ThresholdPortfolio(IntervalPortfolio):
    def __init__(self,
                 thresholds: Thresholds,
                 budget: Union[int, float] = None,
                 fee_rate: Union[int, float] = None,
                 fee_fixed: Union[int, float] = None,
                 name: str = None):
        if isinstance(thresholds, Thresholds):
            super().__init__(thresholds, budget, fee_rate, fee_fixed, name)
        else:
            raise ValueError('thresholds must be Thresholds')
