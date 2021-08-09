from typing import Literal

import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pqr import Factor


@pytest.mark.parametrize(
    ['data', 'method', 'period', 'expected'],
    [
        # test 1
        pytest.param(
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            'static', 1,
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3]]),
            id='static period=1'
        ),
        # test 2
        pytest.param(
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            'dynamic', 1,
            pd.DataFrame([[1, 1, 1],
                          [.5, .5, .5]]),
            id='dynamic period=1'
        ),
        # test 3
        pytest.param(
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            'static', 2,
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2]]),
            id='static period=2'
        ),
        # test 4
        pytest.param(
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            'dynamic', 2,
            pd.DataFrame([[2, 2, 2]]),
            id='dynamic period=2'
        ),
    ]
)
def test_look_back(
        data: pd.DataFrame,
        method: Literal['static', 'dynamic'],
        period: int,
        expected: pd.DataFrame
):
    factor = Factor(data)
    factor.look_back(period, method)
    assert_allclose(factor.data, expected)
