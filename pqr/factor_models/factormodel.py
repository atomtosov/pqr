from abc import ABC, abstractmethod
from typing import Tuple

from pqr.portfolios import Portfolio


class FactorModel(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        ...

    @property
    @abstractmethod
    def portfolios(self) -> Tuple[Portfolio]:
        ...

    def compare_portfolios(self):
        ...
