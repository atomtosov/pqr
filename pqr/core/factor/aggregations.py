from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "Static",
    "Dynamic",
]


@dataclass
class Static:
    def __call__(self, values: pd.DataFrame) -> pd.Series:
        return pd.Series(
            np.apply_along_axis(lambda x: x[0], 0, values.to_numpy()),
            index=values.columns.copy()
        )


@dataclass
class Dynamic:
    def __call__(self, values: pd.DataFrame) -> pd.Series:
        return pd.Series(
            np.apply_along_axis(lambda x: x[-1] / x[0] - 1, 0, values.to_numpy()),
            index=values.columns.copy()
        )
