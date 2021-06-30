import pytest
from numpy.testing import assert_allclose

import numpy as np
import pandas as pd

from pqr.factors.single_factors import SingleFactor
from pqr.factors.multi_factors.multifactor import MultiFactor

test_data = [
    # Single Factors
    # test 1: static single factor looking=1, lag=0
    pytest.param(
        SingleFactor(pd.DataFrame([[1, 1, 1, 1],
                                   [2, 2, 2, 2],
                                   [3, 3, 3, 3],
                                   [4, 4, 4, 4],
                                   [5, 5, 5, 5]]),
                     dynamic=False),
        1,
        0,
        pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                      [1, 1, 1, 1],
                      [2, 2, 2, 2],
                      [3, 3, 3, 3],
                      [4, 4, 4, 4]]),
        id='static single looking=1, lag=0'
    ),
    # test 2: static single factor looking=2, lag=0
    pytest.param(
        SingleFactor(pd.DataFrame([[1, 1, 1, 1],
                                   [2, 2, 2, 2],
                                   [3, 3, 3, 3],
                                   [4, 4, 4, 4],
                                   [5, 5, 5, 5]]),
                     dynamic=False),
        2,
        0,
        pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [1, 1, 1, 1],
                      [2, 2, 2, 2],
                      [3, 3, 3, 3]]),
        id='static single looking=2, lag=0'
    ),
    # test 3: static single factor looking=2, lag=1
    pytest.param(
        SingleFactor(pd.DataFrame([[1, 1, 1, 1],
                                   [2, 2, 2, 2],
                                   [3, 3, 3, 3],
                                   [4, 4, 4, 4],
                                   [5, 5, 5, 5]]),
                     dynamic=False),
        2,
        1,
        pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [1, 1, 1, 1],
                      [2, 2, 2, 2]]),
        id='static single looking=2, lag=1'
    ),
    # test 4: dynamic single factor looking=1, lag=0
    pytest.param(
        SingleFactor(pd.DataFrame([[1, 1, 1, 1],
                                   [2, 2, 2, 2],
                                   [3, 3, 3, 3],
                                   [4, 4, 4, 4],
                                   [5, 5, 5, 5]]),
                     dynamic=True),
        1,
        0,
        pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [1, 1, 1, 1],
                      [1/2, 1/2, 1/2, 1/2],
                      [1/3, 1/3, 1/3, 1/3]]),
        id='dynamic single looking=1, lag=0'
    ),
    # test 5: dynamic single factor looking=2, lag=0
    pytest.param(
        SingleFactor(pd.DataFrame([[1, 1, 1, 1],
                                   [2, 2, 2, 2],
                                   [3, 3, 3, 3],
                                   [4, 4, 4, 4],
                                   [5, 5, 5, 5]]),
                     dynamic=True),
        2,
        0,
        pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [2, 2, 2, 2],
                      [1, 1, 1, 1]]),
        id='dynamic single looking=2, lag=0'
    ),
    # test 6: dynamic single factor looking=2, lag=1
    pytest.param(
        SingleFactor(pd.DataFrame([[1, 1, 1, 1],
                                   [2, 2, 2, 2],
                                   [3, 3, 3, 3],
                                   [4, 4, 4, 4],
                                   [5, 5, 5, 5]]),
                     dynamic=True),
        2,
        1,
        pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [2, 2, 2, 2]]),
        id='dynamic single looking=2, lag=1'
    ),
    # Multi Factors
    # test 1: 2 static factors looking=1, lag=0
    pytest.param(
        MultiFactor(
            [
                SingleFactor(pd.DataFrame([[1, 1, 1, 1],
                                           [2, 2, 2, 2],
                                           [3, 3, 3, 3],
                                           [4, 4, 4, 4],
                                           [5, 5, 5, 5]]),
                             dynamic=False),
                SingleFactor(pd.DataFrame([[1, 1, 1, 1],
                                           [2, 2, 2, 2],
                                           [3, 3, 3, 3],
                                           [4, 4, 4, 4],
                                           [5, 5, 5, 5]]),
                             dynamic=False),
            ]
        ),
        1,
        0,
        (
            pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                          [1, 1, 1, 1],
                          [2, 2, 2, 2],
                          [3, 3, 3, 3],
                          [4, 4, 4, 4]]),
            pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                          [1, 1, 1, 1],
                          [2, 2, 2, 2],
                          [3, 3, 3, 3],
                          [4, 4, 4, 4]])
        ),
        id='2 static multi looking=1, lag=0'
    ),
    # test 2: 2 dynamic factors looking=2, lag=0
    pytest.param(
        MultiFactor(
            [
                SingleFactor(pd.DataFrame([[1, 1, 1, 1],
                                           [2, 2, 2, 2],
                                           [3, 3, 3, 3],
                                           [4, 4, 4, 4],
                                           [5, 5, 5, 5]]),
                             dynamic=True),
                SingleFactor(pd.DataFrame([[1, 1, 1, 1],
                                           [2, 2, 2, 2],
                                           [3, 3, 3, 3],
                                           [4, 4, 4, 4],
                                           [5, 5, 5, 5]]),
                             dynamic=True),
            ]
        ),
        2,
        0,
        (
            pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [2, 2, 2, 2],
                          [1, 1, 1, 1]]),
            pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [2, 2, 2, 2],
                          [1, 1, 1, 1]])
        ),
        id='2 dynamic multi looking=2, lag=0'
    ),
    # test 3: 2 different factors looking=2, lag=1
    pytest.param(
        MultiFactor(
            [
                SingleFactor(pd.DataFrame([[1, 1, 1, 1],
                                           [2, 2, 2, 2],
                                           [3, 3, 3, 3],
                                           [4, 4, 4, 4],
                                           [5, 5, 5, 5]]),
                             dynamic=True),
                SingleFactor(pd.DataFrame([[1, 1, 1, 1],
                                           [2, 2, 2, 2],
                                           [3, 3, 3, 3],
                                           [4, 4, 4, 4],
                                           [5, 5, 5, 5]]),
                             dynamic=False),
            ]
        ),
        2,
        1,
        (
            pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [2, 2, 2, 2]]),
            pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [1, 1, 1, 1],
                          [2, 2, 2, 2]]),
        ),
        id='dynamic+static multi looking=2, lag=1'
    )
]


@pytest.mark.parametrize(
    'factor, looking_period, lag_period, expected',
    test_data
)
def test_single_factor_transform(
        factor,
        looking_period,
        lag_period,
        expected
):
    assert_allclose(
        factor.transform(looking_period, lag_period),
        expected,
        equal_nan=True
    )
