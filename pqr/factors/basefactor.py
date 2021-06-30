from abc import abstractmethod
from typing import Optional


import pandas as pd


class BaseFactor:
    """
    Abstract base class for all kinds of factors.
    """

    dynamic: bool
    bigger_better: Optional[bool]

    def __repr__(self) -> str:
        """
        Dunder/Magic method for fancy printing BaseFactor object in console.
        """

        return f'{self.__class__.__name__}({self._name})'

    @abstractmethod
    def transform(self,
                  looking_period: int,
                  lag_period: int) -> pd.DataFrame:
        """
        Method to transform factor values by looking period and lag period.
        """

    @property
    @abstractmethod
    def dynamic(self) -> bool:
        """
        bool : Logical property, which shows whether absolute values of factor
        are used to make decision or relative (pct changes).
        """

    @property
    @abstractmethod
    def bigger_better(self) -> Optional[bool]:
        """
        bool, optional : Logical property, which shows whether bigger values of
        factor should be considered as better or on the contrary as worse. If
        it is not obvious, higher or lower values of the factor indicate
        a higher attractiveness of the company (e.g. complicated mix of
        factors), can be None.
        """

    @property
    @abstractmethod
    def _name(self) -> str:
        """
        str : Name of factor.
        """
