from typing import Sequence, Tuple

import numpy as np

from ..basefactor import BaseFactor
from ..interfaces import IFactor


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
        periodicity
        name
        factors
    """

    def __init__(self,
                 factors: Sequence[IFactor],
                 name: str = None):
        # dynamic if only one factor is dynamic
        dynamic = any([factor.dynamic for factor in factors])
        # bigger_better if all factors are bigger_better
        bigger_better = all([factor.bigger_better for factor in factors])
        # lower_better if all factors are lower_better
        lower_better = all([not factor.bigger_better for factor in factors])

        # init parent BaseFactor
        BaseFactor.__init__(
            self,
            dynamic,
            # if not bigger better and not lower_better, than None
            bigger_better or (False if lower_better else None),
            factors[0].periodicity.name,
            name
        )

        self.factors = factors

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
            [factor.transform(looking_period, lag_period)
             for factor in self.factors]
        )

    @property
    def factors(self) -> Tuple[IFactor, ...]:
        return self._factors

    @factors.setter
    def factors(self, value: Sequence[IFactor]) -> None:
        if np.all([isinstance(factor, IFactor) for factor in value]):
            self._factors = tuple(value)
        else:
            raise ValueError('all factors must be Factor')
