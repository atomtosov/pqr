from typing import Union

from pqr.utils import HasNameMixin, DataPeriodicity


class BaseFactor(HasNameMixin):
    """
    Abstract base class for Factors
    Inherits from HasNameMixin to be nameable

    Attributes:
        dynamic: bool - is factor dynamic or not, this information is needed
        for future transformation of factor data
        bigger_better: bool | None - is better, when factor value bigger
        (e.g. ROA) or when factor value lower (e.g. P/E); if value is None it
        means that it cannot be said exactly, what is better (used for multi-
        factors)
        periodicity: DataPeriodicity - info about periodicity or discreteness
        of factor data, used for annualization and smth more
        name: str - name of factor

    Raises ValueError if values to be set as attributes are incorrect
    """

    _dynamic: bool
    _bigger_better: Union[bool, None]
    _periodicity: DataPeriodicity

    def __init__(self,
                 dynamic: bool,
                 bigger_better: Union[bool, None],
                 periodicity: str,
                 name: str):
        """
        Initialization of BaseFactor class

        :param dynamic: is factor dynamic or not
        :param bigger_better: is better, when factor value bigger
        :param periodicity: periodicity or discreteness of factor data
        :param name: name of factor
        """
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
            raise ValueError(
                f'periodicity must be one of values: '
                f'{", ".join(list(DataPeriodicity.__members__.keys()))}'
            )
