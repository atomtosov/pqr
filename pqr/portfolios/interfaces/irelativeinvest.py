from abc import abstractmethod


class IRelativeInvest:
    """
    Class for portfolios, investing in stocks only theoretically to compute
    theoretical returns of strategy.
    """

    @abstractmethod
    def invest(self, *args, **kwargs) -> None:
        """
        Method for filling portfolio with relative positions (1/0/-1).
        """
