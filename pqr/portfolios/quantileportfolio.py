from typing import Union

from pqr.base.portfolio import BasePortfolio
from pqr.base.limits import Quantiles


class QuantilePortfolio(BasePortfolio):
    def __init__(
            self,
            lower_quantile: Union[int, float] = 0,
            upper_quantile: Union[int, float] = 1,
            budget: Union[int, float] = None,
            fee_rate: Union[int, float] = None,
            fee_fixed: Union[int, float] = None
    ):
        super().__init__(
            Quantiles(lower_quantile, upper_quantile),
            budget,
            fee_rate,
            fee_fixed
        )
