import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pqr.factors import Factor, select
from pqr.thresholds import Thresholds, Quantiles

test_data = [
    # test 1: factor q(0, 1/2) equal matrices
    pytest.param(
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]]),
        Factor(pd.DataFrame([[1, 2, 3, 4],
                             [2, 4, 6, 8],
                             [6, 8, 12, 16],
                             [12, 16, 24, 32],
                             [24, 32, 72, 96]])),
        Quantiles(0, 1 / 2),
        pd.DataFrame([[True, True, False, False],
                      [True, True, False, False],
                      [True, True, False, False],
                      [True, True, False, False],
                      [True, True, False, False]]),
        id='factor q(0, 1/2) equal matrices'
    ),
    # test 2: factor q(0, 1/2) not equal matrices
    pytest.param(
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]]),
        Factor(pd.DataFrame([[1, 6, 11, 16],
                             [2, 7, 12, 17],
                             [3, 8, 13, 18],
                             [4, 9, 14, 19],
                             [5, 10, 15, 20]]).shift().dropna()),
        Quantiles(0, 1 / 2),
        pd.DataFrame([[True, True, False, False],
                      [True, True, False, False],
                      [True, True, False, False],
                      [True, True, False, False]]),
        id='factor q(0, 1/2) not equal matrices'
    ),
    # test 3: factor q(0, 1/2) not equal matrices + corrupted prices
    pytest.param(
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, np.nan, 1, 1],
                      [np.nan, 1, 1, 1]]),
        Factor(pd.DataFrame([[1, 6, 11, 16],
                             [2, 7, 12, 17],
                             [3, 8, 13, 18],
                             [4, 9, 14, 19],
                             [5, 10, 15, 20]]).shift().dropna()),
        Quantiles(0, 1 / 2),
        pd.DataFrame([[True, True, False, False],
                      [True, True, False, False],
                      [True, False, True, False],
                      [False, True, True, False]]),
        id='factor q(0, 1/2) not equal matrices + corrupted prices'
    ),
]


@pytest.mark.parametrize(
    'prices, factor, thresholds, expected',
    test_data
)
def test_selecting_by_factor(
        prices: pd.DataFrame,
        factor: Factor,
        thresholds: Thresholds,
        expected: pd.DataFrame
):
    assert_allclose(
        select(prices, factor, thresholds),
        expected,
        equal_nan=True
    )
