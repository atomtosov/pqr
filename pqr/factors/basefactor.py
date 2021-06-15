from typing import Union

from pqr.utils import HasNameMixin, DataPeriodicity


class BaseFactor(HasNameMixin):
    _dynamic: bool
    _bigger_better: Union[bool, None]
    _periodicity: DataPeriodicity

    def __init__(self,
                 dynamic: bool,
                 bigger_better: Union[bool, None],
                 periodicity: str,
                 name: str):
        self.dynamic = dynamic
        self.bigger_better = bigger_better
        self.periodicity = periodicity
        super().__init__(name)

    @property
    def dynamic(self) -> bool:
        return self._dynamic

    @dynamic.setter
    def dynamic(self, value: bool) -> None:
        if isinstance(value, bool):
            self._dynamic = value
        else:
            raise ValueError('static must be bool (True or False)')

    @property
    def bigger_better(self) -> Union[bool, None]:
        return self._bigger_better

    @bigger_better.setter
    def bigger_better(self, value: Union[bool, None]) -> None:
        if isinstance(value, bool) or value is None:
            self._bigger_better = value
        else:
            raise ValueError('bigger_better must be bool or None')

    @property
    def periodicity(self):
        return self._periodicity

    @periodicity.setter
    def periodicity(self, value: str) -> None:
        if isinstance(value, str):
            self._periodicity = getattr(DataPeriodicity, value)
        else:
            # TODO: nice error output
            raise ValueError('data_periodicity must be ')
