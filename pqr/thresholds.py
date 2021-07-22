"""
This module contains instruments to store different thresholds expressively.
During the process of building a portfolio you need to impose restrictions on
factors to filter stock universe, from which you pick stocks, and to select
stocks from this pre-filtered stock universe.

Pqr provides 2 specific types of thresholds - quantiles and tops to filtrate
and select any data, as well as simple thresholds. You can inherit from any of
them to create a new type of thresholds and contribute to source code to add
suport of them.
"""


from typing import Union

import dataclasses

import numpy as np


__all__ = [
    'Thresholds',
    'Quantiles',
    'Top',
]


@dataclasses.dataclass(frozen=True)
class Thresholds:
    """
    Class for simple thresholds from lower value to upper.

    These thresholds are used as simple bounds (e.g. trading volume >
    10 000 000$/day).

    Parameters
    ----------
    lower
        Lower threshold.
    upper
        Upper threshold.
    """

    lower: Union[int, float] = -np.inf
    """Lower threshold."""
    upper: Union[int, float] = np.inf
    """Upper threshold."""


@dataclasses.dataclass(frozen=True)
class Quantiles(Thresholds):
    """
    Class for quantile thresholds.

    Quantile thresholds are used to select or filtrate the data from lower
    quantile to upper (e.g. build a portfolio with companies, having the lowest
    30% of P/E on the market).

    Parameters
    ----------
    lower
        Lower quantile.
    upper
        Upper quantile.

    Raises
    ------
    ValueError
        One of quantiles isn't in range [0,1].
    """

    lower: Union[int, float] = 0
    """Lower quantile."""
    upper: Union[int, float] = 1
    """Upper quantile."""

    def __post_init__(self):
        if not (0 <= self.lower <= 1 and 0 <= self.upper <= 1):
            raise ValueError('quantiles must be in range [0;1]')


@dataclasses.dataclass(frozen=True)
class Top(Thresholds):
    """
    Class for thresholds of top levels.

    Top levels are used to select or filtrate any piece of ranked data (e.g.
    pick into portfolio top-10 stocks with largest market capitalization).

    Parameters
    ----------
    lower
        Lower top level.
    upper
        Upper top level.

    Raises
    ------
    TypeError
        Lower or upper boarder is not int.
    ValueError
        One of top levels is less than 1.
    """

    lower: int = 10
    """Lower top level."""
    upper: int = 1
    """Upper top level."""

    def __post_init__(self):
        if not (isinstance(self.lower, int) and isinstance(self.upper, int)):
            raise TypeError('top levels must be int')

        if not (self.lower >= 1 and self.upper >= 1):
            raise ValueError('top levels must be >= 1')
