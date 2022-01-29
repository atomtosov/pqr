__all__ = [
    "filter",
    "look_back_pct_change",
    "look_back_mean",
    "look_back_median",
    "look_back_min",
    "look_back_max",
    "lag",
    "hold",
]

import numpy as np
import pandas as pd

from pqr.utils import align


def filter(
        factor: pd.DataFrame,
        universe: pd.DataFrame,
) -> pd.DataFrame:
    universe, factor = align(universe, factor)
    return pd.DataFrame(
        np.where(universe.to_numpy(bool), factor.to_numpy(float), np.nan),
        index=factor.index.copy(),
        columns=factor.columns.copy(),
    )


def look_back_pct_change(
        factor: pd.DataFrame,
        period: int,
) -> pd.DataFrame:
    factor_array = factor.to_numpy()
    return pd.DataFrame(
        (factor_array[period:] - factor_array[:-period]) / factor_array[:-period],
        index=factor.index[period:].copy(),
        columns=factor.columns.copy()
    )


def look_back_mean(
        factor: pd.DataFrame,
        period: int,
) -> pd.DataFrame:
    return factor.rolling(period + 1, axis=0).mean().iloc[period:]


def look_back_median(
        factor: pd.DataFrame,
        period: int,
) -> pd.DataFrame:
    return factor.rolling(period + 1, axis=0).median().iloc[period:]


def look_back_min(
        factor: pd.DataFrame,
        period: int,
) -> pd.DataFrame:
    return factor.rolling(period + 1, axis=0).min().iloc[period:]


def look_back_max(
        factor: pd.DataFrame,
        period: int,
) -> pd.DataFrame:
    return factor.rolling(period + 1, axis=0).max().iloc[period:]


def lag(
        factor: pd.DataFrame,
        period: int,
) -> pd.DataFrame:
    if period == 0:
        return factor
    elif period > 0:
        return pd.DataFrame(
            factor.to_numpy()[:-period],
            index=factor.index[period:].copy(),
            columns=factor.columns.copy()
        )
    else:
        return pd.DataFrame(
            factor.to_numpy()[-period:],
            index=factor.index[:period].copy(),
            columns=factor.columns.copy()
        )


def hold(
        factor: pd.DataFrame,
        period: int,
) -> pd.DataFrame:
    periods = np.zeros(len(factor), dtype=int)
    update_periods = np.arange(len(factor), step=period)
    periods[update_periods] = update_periods
    update_mask = np.maximum.accumulate(periods[:, np.newaxis], axis=0)

    return pd.DataFrame(
        np.take_along_axis(factor.to_numpy(), update_mask, axis=0),
        index=factor.index.copy(),
        columns=factor.columns.copy()
    )
