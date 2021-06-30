import pytest
from numpy.testing import assert_allclose

import numpy as np
import pandas as pd

from pqr.factors import PickingFactor, \
    InterceptMultiFactor, NSortMultiFactor, WeighMultiFactor
from pqr.intervals import Quantiles

test_data = [
    # Single Factors
    # test 1: static single factor q(0, 1/2), looking=1, lag=0
    pytest.param(
        PickingFactor(pd.DataFrame([[1, 2, 3, 4],
                                    [2, 4, 6, 8],
                                    [6, 8, 12, 16],
                                    [12, 16, 24, 32],
                                    [24, 32, 72, 96]]),
                      dynamic=False),
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]]),
        Quantiles(0, 1/2),
        1,
        0,
        pd.DataFrame([[False, False, False, False],
                      [True, True, False, False],
                      [True, True, False, False],
                      [True, True, False, False],
                      [True, True, False, False]]),
        id='static single q(0, 1/2) looking=1, lag=0'
    ),
    # test 2: dynamic single factor q(0, 1/2), looking=2, lag=1
    pytest.param(
        PickingFactor(pd.DataFrame([[1, 6, 11, 16],
                                    [2, 7, 12, 17],
                                    [3, 8, 13, 18],
                                    [4, 9, 14, 19],
                                    [5, 10, 15, 20]]),
                      dynamic=True),
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]]),
        Quantiles(0, 1/2),
        2,
        1,
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
        PickingFactor(pd.DataFrame([[1, 6, 11, 16],
                                    [2, 7, 12, 17],
                                    [3, 8, 13, 18],
                                    [4, 9, 14, 19],
                                    [5, 10, 15, 20]]),
                      dynamic=False),
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, np.nan, 1, 1],
                      [np.nan, 1, 1, 1]]),
        Quantiles(0, 1/2),
        2,
        0,
        pd.DataFrame([[False, False, False, False],
                      [False, False, False, False],
                      [True, True, False, False],
                      [True, False, True, False],
                      [False, True, True, False]]),
        id='static single q(0, 1/2) looking=2, lag=0 + corrupted prices'
    ),
    # test 4: 2 static multi intercept factor q(0, 1/2) looking=2, lag=0
    pytest.param(
        InterceptMultiFactor(
            [
                PickingFactor(pd.DataFrame([[1, 6, 11, 16],
                                            [2, 7, 12, 17],
                                            [3, 8, 13, 18],
                                            [4, 9, 14, 19],
                                            [5, 10, 15, 20]]),
                              dynamic=False),
                PickingFactor(pd.DataFrame([[11, 6, 1, 16],
                                            [12, 7, 2, 17],
                                            [13, 8, 3, 18],
                                            [14, 9, 4, 19],
                                            [15, 10, 5, 20]]),
                              dynamic=False)
            ]
        ),
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]]),
        Quantiles(0, 1/2),
        2,
        0,
        pd.DataFrame([[False, False, False, False],
                      [False, False, False, False],
                      [False, True, False, False],
                      [False, True, False, False],
                      [False, True, False, False]]),
        id='2 static multi intercept q(0, 1/2) looking=2, lag=0'
    ),
    # test 5: 2 static multi nsort factor q(0, 1/2) looking=2, lag=0
    pytest.param(
        NSortMultiFactor(
            [
                PickingFactor(pd.DataFrame([[1, 6, 11, 16],
                                            [2, 7, 12, 17],
                                            [3, 8, 13, 18],
                                            [4, 9, 14, 19],
                                            [5, 10, 15, 20]]),
                              dynamic=False),
                PickingFactor(pd.DataFrame([[1, 6, 11, 16],
                                            [2, 7, 12, 17],
                                            [3, 8, 13, 18],
                                            [4, 9, 14, 19],
                                            [5, 10, 15, 20]]),
                              dynamic=False)
            ]
        ),
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]]),
        Quantiles(0, 1/2),
        2,
        0,
        pd.DataFrame([[False, False, False, False],
                      [False, False, False, False],
                      [True, False, False, False],
                      [True, False, False, False],
                      [True, False, False, False]]),
        id='2 static multi nsort q(0, 1/2) looking=2, lag=0'
    ),
    # test 6: 2 static multi weigh factor q(0, 1/2) looking=2, lag=0
    pytest.param(
        WeighMultiFactor(
            [
                PickingFactor(pd.DataFrame([[1, 6, 11, 16],
                                            [2, 7, 12, 17],
                                            [3, 8, 13, 18],
                                            [4, 9, 14, 19],
                                            [5, 10, 15, 20]]),
                              dynamic=False),
                PickingFactor(pd.DataFrame([[1, 6, 11, 16],
                                            [2, 7, 12, 17],
                                            [3, 8, 13, 18],
                                            [4, 9, 14, 19],
                                            [5, 10, 15, 20]]),
                              dynamic=False)
            ]
        ),
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, np.nan, 1, 1],
                      [np.nan, 1, 1, 1]]),
        Quantiles(0, 1/2),
        2,
        0,
        pd.DataFrame([[False, False, False, False],
                      [False, False, False, False],
                      [True, True, False, False],
                      [True, False, True, False],
                      [False, True, True, False]]),
        id='2 static multi weigh q(0, 1/2) looking=2, lag=0'
    )
]


@pytest.mark.parametrize(
    'factor, prices, interval, looking_period, lag_period, expected',
    test_data
)
def test_picking_factor(
        factor,
        prices,
        interval,
        looking_period,
        lag_period,
        expected
):
    assert_allclose(
        factor.pick(prices, interval, looking_period, lag_period),
        expected,
        equal_nan=True
    )
