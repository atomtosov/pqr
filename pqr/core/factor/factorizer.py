from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import pandas as pd

from pqr.core.utils import compose
from .factor import Factor

__all__ = [
    "Factorizer",
]

FactorPreprocessor = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class Factorizer:
    preprocessors: Sequence[FactorPreprocessor]

    def __post_init__(self):
        self._preprocessor = compose(*self.preprocessors)

    def __call__(
            self,
            values: pd.DataFrame,
            better: Literal["more", "less"],
    ) -> Factor:
        return Factor(
            self._preprocessor(values),
            better
        )
