from abc import abstractmethod
from typing import Optional

import numpy as np

from pqr.table import Table
from pqr.utils import DataPeriodicity


class BaseFactor(Table):
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
        values
        index
        columns
        name
    """

    dynamic: bool
    bigger_better: Optional[bool]
    periodicity: DataPeriodicity

    def __init__(self,
                 dynamic: bool,
                 bigger_better: Optional[bool],
                 periodicity: str,
                 name: str):
        """
        Initialize BaseFactor instance.
        """

        self.dynamic = dynamic
        self.bigger_better = bigger_better
        self.periodicity = periodicity
        super().__init__(
            np.array([]),
            np.array([]),
            np.array([]),
            name
        )

    @abstractmethod
    def transform(self,
                  looking_period: int = 1,
                  lag_period: int = 0) -> np.ndarray:
        """
        Transform factor values into appropriate for decision-making format.

        Parameters
        ----------
        looking_period : int, default=1
            Period to lookahead in data to transform it.
        lag_period : int, default=0
            Period to lag data to create effect of delayed reaction to factor
            values.

        Returns
        -------
            2-d matrix with shape equal to shape of data with transformed
            factor values. First looking_period+lag_period lines are equal to
            np.nan, because in these moments decision-making is abandoned
            because of lack of data. For dynamic factors one more line is equal
            to np.nan (see above).
        """

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
    def bigger_better(self) -> Optional[bool]:
        return self._bigger_better

    @bigger_better.setter
    def bigger_better(self, value: Optional[bool]) -> None:
        if isinstance(value, bool) or value is None:
            self._bigger_better = value
        else:
            raise ValueError('bigger_better must be bool or None')

    @property
    def periodicity(self) -> DataPeriodicity:
        return self._periodicity

    @periodicity.setter
    def periodicity(self, value: str) -> None:
        if isinstance(value, str):
            self._periodicity = getattr(DataPeriodicity, value)
        else:
            raise ValueError(
                f'periodicity must be one of values: '
                f'{", ".join(DataPeriodicity.__members__.keys())}'
            )
