from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import align, array_to_alike_df_or_series

__all__ = [
    "Universe",
]


class Universe:
    __slots__ = (
        "values",
    )

    def __init__(self, values: pd.DataFrame):
        self.values: pd.DataFrame = values.astype(bool)

    @classmethod
    def from_prices(cls, prices: pd.DataFrame) -> Universe:
        return cls(prices.notnull())

    def filter(self, mask: pd.Series | pd.DataFrame) -> Universe:
        values, mask = align(self.values, mask)

        self.values: pd.DataFrame = array_to_alike_df_or_series(
            np.where(~(mask.to_numpy()), values.to_numpy(), False),
            values
        )

        return self

    def unify(self, *dfs: pd.DataFrame) -> tuple[pd.DataFrame]:
        dfs: list[pd.DataFrame, ...] = list(dfs)
        for i, df in enumerate(dfs):
            df, uni = align(df, self.values)
            dfs[i]: pd.DataFrame = array_to_alike_df_or_series(
                np.where(uni.to_numpy(), df.to_numpy(), np.nan),
                df
            )

        return tuple(dfs)

    def __copy__(self) -> Universe:
        return Universe(self.values)

    def __invert__(self) -> Universe:
        return Universe(
            array_to_alike_df_or_series(
                ~self.values.to_numpy(),
                self.values
            )
        )

    def __or__(self, other: Universe) -> Universe:
        if isinstance(other, Universe):
            self_values, other_values = align(self.values, other.values)
            return Universe(
                array_to_alike_df_or_series(
                    self_values.to_numpy() | other_values.to_numpy(),
                    self_values
                )
            )

        raise NotImplementedError("logical or supported only between 2 universes")

    __ror__ = __or__

    def __ior__(self, other: Universe) -> Universe:
        if isinstance(other, Universe):
            self_values, other_values = align(self.values, other.values)

            self.values = array_to_alike_df_or_series(
                self_values.to_numpy() | other_values,
                self_values
            )

            return self

        raise NotImplementedError("logical or supported only for 2 universes")

    def __and__(self, other: Universe) -> Universe:
        if isinstance(other, Universe):
            self_values, other_values = align(self.values, other.values)
            return Universe(
                array_to_alike_df_or_series(
                    self_values.to_numpy() & other_values.to_numpy(),
                    self_values
                )
            )

        raise NotImplementedError("logical and supported only between 2 universes")

    __rand__ = __and__

    def __iand__(self, other: Universe) -> Universe:
        if isinstance(other, Universe):
            self_values, other_values = align(self.values, other.values)

            self.values = array_to_alike_df_or_series(
                self_values.to_numpy() & other_values,
                self_values
            )

            return self

        raise NotImplementedError("logical and supported only for 2 universes")
