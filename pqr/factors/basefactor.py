from abc import abstractmethod, ABC
from typing import Union

import numpy as np

from pqr.utils import HasNameMixin, DataPeriodicity


class BaseFactor(ABC, HasNameMixin):
    """
    Abstract base class for factors.

    Parameters
    ----------
    dynamic : bool
        Whether factor values should be used to make decisions in absolute form
        or in relative form (percentage changes).
    bigger_better : bool, None
        Whether more factor value, better company or less factor value better
        company. If it equals None, cannot be defined correctly (e.g. intercept
        multi-factor).
    periodicity : str
        Discreteness of factor with respect to one year (e.g. 'monthly' equals
        to 12, because there are 12 trading months in 1 year).
    name : str
        Name of factor.

    Attributes
    ----------
        dynamic
        bigger_better
        periodicity
        name
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
        Initialize BaseFactor instance.
        """

        self.dynamic = dynamic
        self.bigger_better = bigger_better
        self.periodicity = periodicity
        super().__init__(name)

    @abstractmethod
    def transform(self,
                  looking_period: int,
                  lag_period: int) -> np.ndarray:
        """
        Transform factor values into appropriate for decision-making format.
        """
        ...

    @property
    def dynamic(self) -> bool:
        return self._dynamic

    @dynamic.setter
    def dynamic(self, value: bool) -> None:
        if isinstance(value, bool):
            self._dynamic = value
        else:
            raise ValueError('dynamic must be bool')

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
