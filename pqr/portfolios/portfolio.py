from abc import ABC
from typing import Union

from .baseportfolio import BasePortfolio


class Portfolio(ABC, BasePortfolio):
    _budget: Union[int, float, None]
    _fee_rate: Union[int, float, None]
    _fee_fixed: Union[int, float, None]

    def __init__(self,
                 budget: Union[int, float] = None,
                 fee_rate: Union[int, float] = None,
                 fee_fixed: Union[int, float] = None,
                 name: str = None):
        super().__init__()
        BasePortfolio.__init__(self, name)

        self.budget = budget
        self.fee_rate = fee_rate
        self.fee_fixed = fee_fixed

    @property
    def budget(self) -> Union[int, float, None]:
        return self._budget

    @budget.setter
    def budget(self, value: Union[int, float, None]) -> None:
        if (isinstance(value, (int, float)) and value > 0) or value is None:
            self._budget = value
        else:
            raise ValueError('budget must be int or float > 0 or None')

    @property
    def fee_rate(self) -> Union[int, float, None]:
        return self._fee_rate

    @fee_rate.setter
    def fee_rate(self, value: Union[int, float, None]) -> None:
        if isinstance(value, (int, float)) or value is None:
            self._fee_rate = value
        else:
            raise ValueError('fee_rate must be int or float or None')

    @property
    def fee_fixed(self) -> Union[int, float, None]:
        return self._fee_fixed

    @fee_fixed.setter
    def fee_fixed(self, value: Union[int, float, None]) -> None:
        if isinstance(value, (int, float)) or value is None:
            self._fee_fixed = value
        else:
            raise ValueError('fee_fixed must be int or float or None')
