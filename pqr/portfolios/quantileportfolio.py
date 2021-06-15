from typing import Union

from .intervalportfolio import IntervalPortfolio
from pqr.utils import Quantiles


class QuantilePortfolio(IntervalPortfolio):
    def __init__(self,
                 quantiles: Quantiles,
                 budget: Union[int, float] = None,
                 fee_rate: Union[int, float] = None,
                 fee_fixed: Union[int, float] = None,
                 name: str = None):
        if isinstance(quantiles, Quantiles):
            super().__init__(quantiles, budget, fee_rate, fee_fixed, name)
        else:
            raise ValueError('quantiles must be Quantiles')
