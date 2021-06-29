from abc import abstractmethod
from typing import Optional


import pandas as pd


class BaseFactor:
    """
    Abstract base class for factors.

    Attributes
    ----------
    dynamic : bool
        Whether factor values should be used to make decisions in absolute form
        or in relative form (percentage changes).
    bigger_better : bool, None
        Whether more factor value, better company or less factor value better
        company. If it equals None, cannot be defined correctly (e.g. intercept
        multi-factor).
    """

    dynamic: bool
    bigger_better: Optional[bool]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._name})'

    @abstractmethod
    def transform(self,
                  looking_period: int,
                  lag_period: int) -> pd.DataFrame:
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
    @abstractmethod
    def dynamic(self) -> bool:
        ...

    @property
    @abstractmethod
    def bigger_better(self) -> Optional[bool]:
        ...

    @property
    @abstractmethod
    def _name(self) -> str:
        ...
