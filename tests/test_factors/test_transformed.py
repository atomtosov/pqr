import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pqr.factors import factorize

test_data = [
    # Single Factors
    # test 1: static single factor looking=1, lag=0
    pytest.param(
        factorize(pd.DataFrame([[1, 1, 1, 1],
                             [2, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4],
                             [5, 5, 5, 5]]),
               dynamic=False),
        1, 0, 1,
        pd.DataFrame([[1, 1, 1, 1],
                      [2, 2, 2, 2],
                      [3, 3, 3, 3],
                      [4, 4, 4, 4]]),
        id='static single looking=1, lag=0'
    ),
    # test 2: static single factor looking=2, lag=0
    pytest.param(
        factorize(pd.DataFrame([[1, 1, 1, 1],
                             [2, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4],
                             [5, 5, 5, 5]]),
               dynamic=False),
        2, 0, 1,
        pd.DataFrame([[1, 1, 1, 1],
                      [2, 2, 2, 2],
                      [3, 3, 3, 3]]),
        id='static single looking=2, lag=0'
    ),
    # test 3: static single factor looking=2, lag=1
    pytest.param(
        factorize(pd.DataFrame([[1, 1, 1, 1],
                             [2, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4],
                             [5, 5, 5, 5]]),
               dynamic=False),
        2, 1, 1,
        pd.DataFrame([[1, 1, 1, 1],
                      [2, 2, 2, 2]]),
        id='static single looking=2, lag=1'
    ),
    # test 4: dynamic single factor looking=1, lag=0
    pytest.param(
        factorize(pd.DataFrame([[1, 1, 1, 1],
                             [2, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4],
                             [5, 5, 5, 5]]),
               dynamic=True),
        1, 0, 1,
        pd.DataFrame([[1, 1, 1, 1],
                      [1 / 2, 1 / 2, 1 / 2, 1 / 2],
                      [1 / 3, 1 / 3, 1 / 3, 1 / 3]]),
        id='dynamic single looking=1, lag=0'
    ),
    # test 5: dynamic single factor looking=2, lag=0
    pytest.param(
        factorize(pd.DataFrame([[1, 1, 1, 1],
                             [2, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4],
                             [5, 5, 5, 5]]),
               dynamic=True),
        2, 0, 1,
        pd.DataFrame([[2, 2, 2, 2],
                      [1, 1, 1, 1]]),
        id='dynamic single looking=2, lag=0'
    ),
    # test 6: dynamic single factor looking=2, lag=1
    pytest.param(
        factorize(pd.DataFrame([[1, 1, 1, 1],
                             [2, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4],
                             [5, 5, 5, 5]]),
               is_dynamic=True),
        2, 1, 1,
        pd.DataFrame([[2, 2, 2, 2]]),
        id='dynamic single looking=2, lag=1'
    ),
]


@pytest.mark.parametrize(
    'factor, looking_period, lag_period, holding_period, expected',
    test_data
)
def test_single_factor_transform(
        factor: pd.DataFrame,
        looking_period: int,
        lag_period: int,
        holding_period: int,
        expected: pd.DataFrame
):
    assert_allclose(
        transform(factor, looking_period, lag_period, holding_period).data,
        expected,
        equal_nan=True
    )
