import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pqr import Factor, Portfolio


@pytest.mark.parametrize(
    ['data', 'picks', 'expected'],
    [
        # test 1
        pytest.param(
            pd.DataFrame([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [10, 11, 12]]),
            pd.DataFrame([[True, False, False],
                          [True, False, False],
                          [True, False, False],
                          [True, False, False]]),
            pd.DataFrame([[1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0]]),
            id='all weights are equal to 1'
        ),
        # test 2
        pytest.param(
            pd.DataFrame([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [10, 11, 12]]),
            pd.DataFrame([[True, True, False],
                          [True, True, False],
                          [True, True, False],
                          [True, True, False]]),
            pd.DataFrame([[1 / 3, 2 / 3, 0],
                          [4 / 9, 5 / 9, 0],
                          [7 / 15, 8 / 15, 0],
                          [10 / 21, 11 / 21, 0]]),
            id='different weights'
        ),
        # test 3
        pytest.param(
            pd.DataFrame([[np.nan, 2, 3],
                          [4, np.nan, 6],
                          [7, 8, 9],
                          [np.nan, np.nan, 12]]),
            pd.DataFrame([[True, True, False],
                          [True, True, False],
                          [True, True, False],
                          [True, True, False]]),
            pd.DataFrame([[0, 1, 0],
                          [1, 0, 0],
                          [7 / 15, 8 / 15, 0],
                          [0, 0, 0]]),
            id='different weights with nans'
        ),
    ]
)
def test_weighing_by_factor(
        data: pd.DataFrame,
        picks: pd.DataFrame,
        expected: pd.DataFrame
):
    portfolio = Portfolio()
    portfolio.picks = picks
    factor = Factor(data)
    portfolio.weigh_by_factor(factor)
    assert_allclose(portfolio.weights, expected)
