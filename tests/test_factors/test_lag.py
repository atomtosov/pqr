import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pqr import Factor


@pytest.mark.parametrize(
    ['data', 'period', 'expected'],
    [
        # test 1
        pytest.param(
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            0,
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            id='period=0'
        ),
        # test 2
        pytest.param(
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            1,
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3]]),
            id='period=1'
        ),
        # test 3
        pytest.param(
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            3,
            pd.DataFrame([[1, 1, 1]]),
            id='period=3'
        ),
    ]
)
def test_lag(
        data: pd.DataFrame,
        period: int,
        expected: pd.DataFrame
):
    factor = Factor(data)
    factor.lag(period)
    assert_allclose(factor.data, expected)
