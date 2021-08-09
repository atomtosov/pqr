from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pqr import Factor, Portfolio


@pytest.mark.parametrize(
    ['data', 'target', 'weights', 'expected'],
    [
        # test 1
        pytest.param(
            pd.DataFrame([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [10, 11, 12]]),
            10,
            pd.DataFrame([[1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0]]),
            pd.DataFrame([[0.1, 0, 0],
                          [0.4, 0, 0],
                          [0.7, 0, 0],
                          [1, 0, 0]]),
            id='simple scaling'
        ),
        # test 2
        pytest.param(
            pd.DataFrame([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [10, 11, 12]]),
            5,
            pd.DataFrame([[1 / 3, 2 / 3, 0],
                          [4 / 9, 5 / 9, 0],
                          [7 / 15, 8 / 15, 0],
                          [10 / 21, 11 / 21, 0]]),
            pd.DataFrame([[1 / 15, 4 / 15, 0],
                          [16 / 45, 5 / 9, 0],
                          [49 / 75, 64 / 75, 0],
                          [20 / 21, 121 / 105, 0]]),
            id='different scales'
        ),
        # test 3
        pytest.param(
            pd.DataFrame([[np.nan, 2, 3],
                          [4, np.nan, 6],
                          [7, 8, 9],
                          [np.nan, np.nan, 12]]),
            10,
            pd.DataFrame([[1, 0, 0],
                          [1, 0, 0],
                          [7 / 15, 8 / 15, 0],
                          [0, 0, 0]]),
            pd.DataFrame([[0, 0, 0],
                          [0.4, 0, 0],
                          [49 / 150, 64 / 150, 0],
                          [0, 0, 0]]),
            id='different scales with nans'
        ),
    ]
)
def test_scaling_by_factor(
        data: pd.DataFrame,
        weights: pd.DataFrame,
        target: int | float,
        expected: pd.DataFrame
):
    portfolio = Portfolio()
    portfolio.weights = weights
    factor = Factor(data)
    portfolio.scale_weights_by_factor(factor, target)
    assert_allclose(portfolio.weights, expected)
