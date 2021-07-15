import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pqr.factors import Factor, pick
from pqr.thresholds import Thresholds, Quantiles

test_data = [
    # test 1: static single factor q(0, 1/2), looking=1, lag=0
    pytest.param(
        Factor(pd.DataFrame([[1, 2, 3, 4],
                             [2, 4, 6, 8],
                             [6, 8, 12, 16],
                             [12, 16, 24, 32],
                             [24, 32, 72, 96]]),
               dynamic=False),
        Quantiles(0, 1 / 2),
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]]),
        1, 0, 1,
        pd.DataFrame([[False, False, False, False],
                      [True, True, False, False],
                      [True, True, False, False],
                      [True, True, False, False],
                      [True, True, False, False]]),
        id='static single q(0, 1/2) looking=1, lag=0'
    ),
    # test 2: dynamic single factor q(0, 1/2), looking=2, lag=1
    pytest.param(
        Factor(pd.DataFrame([[1, 6, 11, 16],
                             [2, 7, 12, 17],
                             [3, 8, 13, 18],
                             [4, 9, 14, 19],
                             [5, 10, 15, 20]]),
               dynamic=True),
        Quantiles(0, 1 / 2),
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]]),
        2, 1, 1,
        pd.DataFrame([[False, False, False, False],
                      [False, False, False, False],
                      [False, False, False, False],
                      [False, False, False, False],
                      [False, False, True, True]]),
        id='dynamic single q(0, 1/2) looking=2, lag=1'
    ),
    # test 3: static single factor q(0, 1/2) looking=2, lag=0
    # + corrupted prices
    pytest.param(
        Factor(pd.DataFrame([[1, 6, 11, 16],
                             [2, 7, 12, 17],
                             [3, 8, 13, 18],
                             [4, 9, 14, 19],
                             [5, 10, 15, 20]]),
               dynamic=False),
        Quantiles(0, 1 / 2),
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, np.nan, 1, 1],
                      [np.nan, 1, 1, 1]]),
        2, 0, 1,
        pd.DataFrame([[False, False, False, False],
                      [False, False, False, False],
                      [True, True, False, False],
                      [True, False, True, False],
                      [False, True, True, False]]),
        id='static single q(0, 1/2) looking=2, lag=0 + corrupted prices'
    ),
]


@pytest.mark.parametrize(
    'factor, thresholds, prices, '
    'looking_period, lag_period, holding_period, '
    'expected',
    test_data
)
def test_picking_by_factor(
        factor: Factor,
        thresholds: Thresholds,
        prices: pd.DataFrame,
        looking_period: int,
        lag_period: int,
        holding_period: int,
        expected: pd.DataFrame
):
    assert_allclose(
        pick(prices, factor, thresholds, looking_period, lag_period,
             holding_period),
        expected,
        equal_nan=True
    )
