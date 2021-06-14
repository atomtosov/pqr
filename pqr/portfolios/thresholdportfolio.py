from typing import Union

import numpy as np

from pqr.base.portfolio import BasePortfolio
from pqr.base.limits import Thresholds


class ThresholdPortfolio(BasePortfolio):
    def __init__(
            self,
            lower_threshold: Union[int, float] = -np.inf,
            upper_threshold: Union[int, float] = np.inf,
            budget: Union[int, float] = None,
            fee_rate: Union[int, float] = None,
            fee_fixed: Union[int, float] = None
    ):
        super().__init__(
            Thresholds(lower_threshold, upper_threshold),
            budget,
            fee_rate,
            fee_fixed
        )
