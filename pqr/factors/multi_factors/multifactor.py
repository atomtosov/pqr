from typing import Sequence, Tuple, Optional

import numpy as np

from ..basefactor import BaseFactor


class MultiFactor(BaseFactor):
    """
    Abstract base class for multi-factors, which consist of single-factors.

    Parameters
    ----------
    factors : sequence of IFactor
        Sequence of factors.
    name : str, optional
        Name of factor.

    Attributes
    ----------
        dynamic
        bigger_better
        name
        factors
    """

    def __init__(self,
                 factors: Sequence[BaseFactor],
                 name: str = ''):
        if isinstance(name, str):
            self._name = name
        else:
            raise ValueError('name must be str')

        if np.all([isinstance(factor, BaseFactor) for factor in factors]):
            self._factors = tuple(factors)
        else:
            raise ValueError('all factors must be Factor')

    def transform(self,
                  looking_period: int = 1,
                  lag_period: int = 0) -> np.ndarray:
        """
        Transform factor values into appropriate for decision-making format.

        All factors are transformed by the same looking and lag periods and
        gathered into one array.

        Parameters
        ----------
        looking_period : int, default=1
            Period to lookahead in factor values to transform it.
        lag_period : int, default=0
            Period to lag factor values to create effect of delayed reaction to
            factor values.

        Returns
        -------
            3-d matrix with shape equal to quantity of factors and shape of
            factor values with transformed factor values. First
            looking_period+lag_period lines  are equal to np.nan, because in
            these moments decision-making is abandoned because of lack of data.
            For dynamic factors one more line is equal to np.nan (see above).
        """

        return np.array(
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
    def name(self):
        return self._name
