import pandas as pd
import numpy as np
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
        data,
        method,
        period,
        expected
):
    factor = Factor(data)
    factor.look_back(period, method)
    assert_allclose(factor.data, expected)


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
def test_hold(
        data: pd.DataFrame,
        period: int,
        expected: pd.DataFrame
):
    factor = Factor(data)
    factor.hold(period)
    assert_allclose(factor.data, expected, equal_nan=True)


@pytest.mark.parametrize(
    ['data', 'mask', 'expected'],
    [
        # test 1
        pytest.param(
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            pd.DataFrame([[True, False, True],
                          [False, True, False],
                          [True, True, True],
                          [False, False, False]]),
            pd.DataFrame([[1, np.nan, 1],
                          [np.nan, 2, np.nan],
                          [3, 3, 3],
                          [np.nan, np.nan, np.nan]]),
            id='mask\'s shape = factor\'s shapes'
        ),
        # test 2
        pytest.param(
            pd.DataFrame([[2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]], index=[1, 2, 3]),
            pd.DataFrame([[True, False, True],
                          [False, True, False],
                          [True, True, True],
                          [False, False, False]]),
            pd.DataFrame([[np.nan, 2, np.nan],
                          [3, 3, 3],
                          [np.nan, np.nan, np.nan]]),
            id='mask\'s shape > factor\'s shape'
        ),
        # test 3
        pytest.param(
            pd.DataFrame([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]]),
            pd.DataFrame([[False, True, False],
                          [True, True, True],
                          [False, False, False]], index=[1, 2, 3]),
            pd.DataFrame([[np.nan, 2, np.nan],
                          [3, 3, 3],
                          [np.nan, np.nan, np.nan]]),
            id='mask\'s shape < factor\'s shape'
        ),
    ]
)
def test_filter(
        data: pd.DataFrame,
        mask: pd.DataFrame,
        expected: pd.DataFrame
):
    factor = Factor(data)
    factor.filter(mask)
    assert_allclose(factor.data, expected, equal_nan=True)
