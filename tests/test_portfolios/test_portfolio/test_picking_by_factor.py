from __future__ import annotations

from typing import Tuple, Literal

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pqr import Factor, Portfolio


@pytest.mark.parametrize(
    ['data', 'thresholds', 'method', 'expected'],
    [
        # test 1
        pytest.param(
            pd.DataFrame([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [10, 11, 12]]),
            (0, 1/3), 'quantile',
            pd.DataFrame([[True, False, False],
                          [True, False, False],
                          [True, False, False],
                          [True, False, False]]),
            id='q(0, 1/3)'
        ),
        # test 2
        pytest.param(
            pd.DataFrame([[np.nan, 2, 3],
                          [4, np.nan, 6],
                          [7, 8, 9],
                          [np.nan, np.nan, 12]]),
            (0, 1/2), 'quantile',
            pd.DataFrame([[False, True, False],
                          [True, False, False],
                          [True, True, False],
                          [False, False, True]]),
            id='q(0, 1/2) with nans'
        ),
        # test 3
        pytest.param(
            pd.DataFrame([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [10, 11, 12]]),
            (1, 2), 'top',
            pd.DataFrame([[False, True, True],
                          [False, True, True],
                          [False, True, True],
                          [False, True, True]]),
            id='top(1, 2)'
        ),
        # test 4
        pytest.param(
            pd.DataFrame([[np.nan, 2, 3],
                          [4, np.nan, 6],
                          [7, 8, 9],
                          [np.nan, np.nan, 12]]),
            (1, 2), 'top',
            pd.DataFrame([[False, True, True],
                          [True, False, True],
                          [False, True, True],
                          [False, False, True]]),
            id='top(1, 2) with nans'
        ),
        # test 5
        pytest.param(
            pd.DataFrame([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [10, 11, 12]]),
            (2, 8), 'time-series',
            pd.DataFrame([[False, True, True],
                          [True, True, True],
                          [True, True, False],
                          [False, False, False]]),
            id='time-series(2, 8)'
        ),
        # test 6
        pytest.param(
            pd.DataFrame([[np.nan, 2, 3],
                          [4, np.nan, 6],
                          [7, 8, 9],
                          [np.nan, np.nan, 12]]),
            (2, 8), 'time-series',
            pd.DataFrame([[False, True, True],
                          [True, False, True],
                          [True, True, False],
                          [False, False, False]]),
            id='time-series(2, 8) with nans'
        ),
    ]
)
def test_picking_by_factor(
        data: pd.DataFrame,
        thresholds: Tuple[int | float, int | float],
        method: Literal['quantile', 'top', 'time-series'],
        expected: pd.DataFrame
):
    portfolio = Portfolio()
    factor = Factor(data)
    portfolio.pick_stocks_by_factor(factor, thresholds, method)
    assert_allclose(portfolio.picks, expected)
