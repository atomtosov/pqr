from typing import Optional
import pandas as pd

from .baseportfolio import BasePortfolio
from .interfaces import IWMLPortfolio, IPortfolio
from pqr.benchmarks.interfaces import IBenchmark


class WMLPortfolio(BasePortfolio, IWMLPortfolio):
    """
    Class for Winners-minus-Losers (WML) portfolios.
    """

    def __init__(self):
        """
        Initialize WMLPortfolio instance.
        """

        self._positions = pd.DataFrame()
        self._returns = pd.Series()

        self._winners = None
        self._losers = None

    def invest(self,
               winners: IPortfolio,
               losers: IPortfolio) -> None:
        """
        Simulating theoretical WML-portfolio (all short-sells are available).

        Parameters
        ----------
        winners : BasePortfolio
            Portfolio of "winners" - stocks with the best values of factor.
        losers : BasePortfolio
            Portfolio of "losers" - stocks with the worst values of factor.
        """

        self._winners = winners
        self._losers = losers

    @property
    def positions(self) -> pd.DataFrame:
        return self.winners.positions - self.losers.positions

    @property
    def returns(self) -> pd.Series:
        return self.winners.returns - self.losers.returns

    @property
    def benchmark(self) -> Optional[IBenchmark]:
        return self.winners.benchmark

    @property
    def shift(self) -> int:
        return self.winners.shift

    @property
    def _name(self) -> str:
        return ''

    @property
    def winners(self) -> IPortfolio:
        """
        IPortfolio : Portfolio of stocks with the best values of factor.
        """

        return self._winners

    @property
    def losers(self) -> IPortfolio:
        """
        IPortfolio : Portfolio of stocks with the worst values of factor.
        """

        return self._losers
