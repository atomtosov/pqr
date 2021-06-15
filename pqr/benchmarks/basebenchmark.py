from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from pqr.utils import HasNameMixin, HasIndexMixin


class BaseBenchmark(ABC, HasNameMixin, HasIndexMixin):
    def __init__(self, name: str = None):
        HasNameMixin.__init__(self, name)
        HasIndexMixin.__init__(self)

    @abstractmethod
    def _calc_returns(self) -> np.ndarray:
        ...

    @property
    def returns(self) -> np.ndarray:
        return self._calc_returns()

    def calc_cumulative_returns(self, shift: int = 0) -> np.ndarray:
        if not isinstance(shift, int) or shift < 0:
            raise ValueError('shift must be int > 0')
        returns = self._calc_returns()
        returns[:shift+1] = np.nan
        return np.nancumprod(returns + 1) - 1

    @property
    def cumulative_returns(self):
        return self.calc_cumulative_returns()

    @property
    def total_return(self):
        return self.cumulative_returns[-1] * 100

    def plot_cumulative_returns(self, shift: int = 0):
        if not isinstance(shift, int) or shift < 0:
            raise ValueError('shift must be int > 0')
        plt.plot(
            self._index,
            self.calc_cumulative_returns(shift),
            label=repr(self)
        )
