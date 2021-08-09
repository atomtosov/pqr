import numpy as np
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
            1,
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            id='period=1'
        ),
        # test 2
        pytest.param(
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            2,
            pd.DataFrame([[1, 1, 1],
                          [1, 1, 1],
                          [3, 3, 3],
                          [3, 3, 3]]),
            id='period=2'
        ),
        # test 3
        pytest.param(
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            3,
            pd.DataFrame([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1],
                          [4, 4, 4]]),
            id='period=3'
        ),
        # test 4
        pytest.param(
            pd.DataFrame([[1, 1, np.nan],
                          [2, 2, 2],
                          [3, 3, 3],
                          [np.nan, 4, 4]]),
            2,
            pd.DataFrame([[1, 1, np.nan],
                          [1, 1, np.nan],
                          [3, 3, 3],
                          [3, 3, 3]]),
            id='period=2 with nans'
        ),
    ]
)
def test_fill_forward(
        data: pd.DataFrame,
        period: int,
        expected: pd.DataFrame
):
    factor = Factor(data)
    factor.hold(period)
    assert_allclose(factor.data, expected, equal_nan=True)
