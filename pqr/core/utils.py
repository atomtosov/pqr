import functools as ft
from typing import TypeVar, Callable

import numpy as np
import pandas as pd

T = TypeVar("T")
ComposableFunction = Callable[[T], T]


def compose(*functions: ComposableFunction) -> ComposableFunction:
    return ft.reduce(
        lambda f, g: lambda x: g(f(x)),
        functions
    )


def normalize(weights: pd.DataFrame) -> pd.DataFrame:
    weights_array = weights.to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.nan_to_num(
            weights_array / np.nansum(weights_array, axis=1, keepdims=True),
            nan=0, neginf=0, posinf=0, copy=False
        )

    return pd.DataFrame(
        w,
        index=weights.index.copy(),
        columns=weights.columns.copy()
    )
