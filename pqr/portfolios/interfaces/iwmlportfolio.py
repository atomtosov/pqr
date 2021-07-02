from typing import Optional
from abc import abstractmethod

from .iportfolio import IPortfolio
from pqr.benchmarks.interfaces import IBenchmark


class IWMLPortfolio:
    """
    Class-interface for WML (winners-minus-losers) portfolios.
    """

    @abstractmethod
    def invest(self,
               winners: IPortfolio,
               losers: IPortfolio) -> None:
        """
        Method for filling portfolio with relative positions (1/0/-1).
        """
