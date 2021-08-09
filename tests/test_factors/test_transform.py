from typing import Literal

import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pqr import Factor


@pytest.mark.parametrize(
    ['data', 'method', 'looking_back_period', 'lag_period', 'holding_period'],
    [
        # test 1
        pytest.param(
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            'static', 1, 0, 1,
            id='static 1-0-1'
        ),
        # test 2
        pytest.param(
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            'dynamic', 1, 0, 1,
            id='dynamic 1-0-1'
        ),
        # test 3
        pytest.param(
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            'static', 2, 0, 2,
            id='static 2-0-2'
        ),
        # test 4
        pytest.param(
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            'dynamic', 2, 0, 2,
            id='dynamic 2-0-2'
        ),
        # test 3
        pytest.param(
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            'static', 2, 1, 1,
            id='static 2-1-1'
        ),
    ]
)
def test_transform(
        data: pd.DataFrame,
        method: Literal['static', 'dynamic'],
        looking_back_period: int,
        lag_period: int,
        holding_period: int
):
    factor = Factor(data)
    factor.transform(looking_back_period, method,
                     lag_period, holding_period)

    factor_test = Factor(data)
    (factor_test.look_back(looking_back_period, method))
    assert_allclose(factor.data, factor_test.data)
