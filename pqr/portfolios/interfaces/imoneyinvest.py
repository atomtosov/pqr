from abc import abstractmethod


class IMoneyInvest:
    @abstractmethod
    def invest_cash(self, *args, **kwargs) -> None:
        ...
