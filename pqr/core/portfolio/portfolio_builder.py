from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from pqr.core.utils import compose
from pqr.utils import align
from .portfolio import Portfolio

__all__ = [
    "PortfolioBuilder",
]

PortfolioBuildingStep = Callable[[Portfolio], Portfolio]


@dataclass
class PortfolioBuilder:
    building_steps: Sequence[PortfolioBuildingStep]

    def __post_init__(self):
        self._builder = compose(*self.building_steps)

    def __call__(
            self,
            longs: Optional[pd.DataFrame] = None,
            shorts: Optional[pd.DataFrame] = None,
            name: Optional[str] = None
    ) -> Portfolio:
        if longs is None and shorts is None:
            raise ValueError("either longs or shorts must be specified")

        elif longs is not None and shorts is not None:  # long-short
            longs, shorts = align(longs, shorts)
            picks = pd.DataFrame(
                longs.to_numpy(dtype=np.int8) - shorts.to_numpy(dtype=np.int8),
                index=longs.index.copy(),
                columns=longs.columns.copy()
            )
        elif longs is not None:  # long-only
            picks = pd.DataFrame(
                longs.to_numpy(dtype=np.int8),
                index=longs.index.copy(),
                columns=longs.columns.copy()
            )
        else:  # short-only
            picks = pd.DataFrame(
                -shorts.to_numpy(dtype=np.int8),
                index=shorts.index.copy(),
                columns=shorts.columns.copy()
            )

        portfolio = Portfolio(picks, name)

        return self._builder(portfolio)
