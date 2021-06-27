from abc import abstractmethod


class IRelativeInvest:
    @abstractmethod
    def invest(self, *args, **kwargs) -> None:
        ...
