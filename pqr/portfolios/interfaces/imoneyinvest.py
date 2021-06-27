from abc import abstractmethod


class IMoneyInvest:
    """
    Class-interface for portfolios, investing real money with rebalancing of
    porftolios.
    """

    @abstractmethod
    def invest_cash(self, *args, **kwargs) -> None:
        """
        Method for investing with cash.
        """
