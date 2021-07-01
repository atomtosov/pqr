from typing import Sequence, Tuple, Optional

import numpy as np
import pandas as pd

from ..basefactor import BaseFactor


class MultiFactor(BaseFactor):
    """
    Abstract base class for multi-factors, consisting of more than 1 factor.
    """

    def __init__(self,
                 factors: Sequence[BaseFactor],
                 name: str = ''):
        """
        Initialize MultiFactor instance.

        Parameters
        ----------
        factors : sequence of IFactor
            Sequence of factors. Must contain at least 2 factors. Can include
            not only single-factors, but also multi-factors.
        name : str, optional
            Name of factor.

        Raises
        ------
        TypeError
            Any of factors is not a BaseFactor.
        ValueError
            Sequence of factors contains less than 2 elements.
        """

        if not np.all([isinstance(factor, BaseFactor) for factor in factors]):
            raise TypeError('all factors must be BaseFactor')
        elif len(factors) <= 1:
            raise ValueError('sequence of factors must contain at least 2'
                             'factors')
        else:
            self._factors = tuple(factors)

        if isinstance(name, str):
            self.__name = name
        else:
            raise ValueError('name must be str')

    def transform(self,
                  looking_period: int = 1,
                  lag_period: int = 0) -> Tuple[pd.DataFrame, ...]:
        """
        Transform factor values into appropriate for decision-making format.

        Simply transforms every factor independently. There is some kind of a
        conflict between static and dynamic factors: if a multi-factor includes
        both of them, static factors have in general one more period of
        observations, but for decision-making it is not used.

        Parameters
        ----------
        looking_period : int, default=1
            Period to lookahead in factor values to transform it.
        lag_period : int, default=0
            Period to lag factor values to create effect of delayed reaction to
            factor values.

        Returns
        -------
        tuple[pd.DataFrame, ...]
            Tuple of dataframes (or tuples of dataframes if a multi-factor also
            include other multi-factors) with transformed views of each factor.
        """

        return tuple(
            [
                factor.transform(looking_period, lag_period)
                for factor in self.factors
            ]
        )

    @property
    def factors(self) -> Tuple[BaseFactor, ...]:
        return self._factors

    @property
    def dynamic(self) -> bool:
        return np.any([factor.dynamic for factor in self.factors])

    @property
    def bigger_better(self) -> Optional[bool]:
        bigger_better = np.all([factor.bigger_better
                                for factor in self.factors])
        lower_better = np.all([not factor.bigger_better
                               for factor in self.factors])
        return bigger_better or lower_better or None

    @property
    def _name(self):
        return self.__name
