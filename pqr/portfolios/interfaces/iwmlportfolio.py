from abc import abstractmethod

from .iportfolio import IPortfolio


class IWMLPortfolio:
    """
    Interface for WML (winners-minus-losers) portfolios: pick stocks from
    choice of winners' portfolio, and sell short choice of losers' portfolio.
    """

    @abstractmethod
    def invest(self,
               winners: IPortfolio,
               losers: IPortfolio) -> None:
        """
        Method for filling positions by 2 factor portfolios.
        """
