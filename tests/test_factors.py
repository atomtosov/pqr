import pytest
from numpy.testing import assert_allclose

import pandas as pd
import numpy as np

from pqr import Factor


@pytest.fixture
def simple_factor():
    return Factor(pd.DataFrame([[1, 1, 1],
                                [2, 2, 2],
                                [3, 3, 3],
                                [4, 4, 4]]))


class TestLookBack:
    @pytest.mark.parametrize('period, expected',
        [
            pytest.param(1, pd.DataFrame([[1, 1, 1], 
                                          [2, 2, 2], 
                                          [3, 3, 3]])),
            pytest.param(2, pd.DataFrame([[1, 1, 1], 
                                          [2, 2, 2]])),
        ]
    )
    def test_static(self, simple_factor, period, expected):
        simple_factor.look_back(period, 'static')
        assert_allclose(simple_factor.data, expected, equal_nan=True)

    @pytest.mark.parametrize('period, expected',
        [
            pytest.param(1, pd.DataFrame([[1, 1, 1], 
                                          [.5, .5, .5]])),
            pytest.param(2, pd.DataFrame([[2, 2, 2]])),
        ]
    )
    def test_dynamic(self, simple_factor, period, expected):
        simple_factor.look_back(period, 'dynamic')
        assert_allclose(simple_factor.data, expected, equal_nan=True)


@pytest.mark.parametrize('period, expected',
    [
        pytest.param(0, pd.DataFrame([[1, 1, 1], 
                                      [2, 2, 2], 
                                      [3, 3, 3],
                                      [4, 4, 4]])),
        pytest.param(1, pd.DataFrame([[1, 1, 1], 
                                      [2, 2, 2], 
                                      [3, 3, 3]])),
        pytest.param(3, pd.DataFrame([[1, 1, 1]])),
    ]
)
def test_lag(simple_factor, period, expected):
    simple_factor.lag(period)
    assert_allclose(simple_factor.data, expected, equal_nan=True)


@pytest.mark.parametrize('period, expected',
    [
        pytest.param(1, pd.DataFrame([[1, 1, 1],
                                      [2, 2, 2],
                                      [3, 3, 3],
                                      [4, 4, 4]])),
        pytest.param(2, pd.DataFrame([[1, 1, 1],
                                      [1, 1, 1],
                                      [3, 3, 3],
                                      [3, 3, 3]])),
        pytest.param(3, pd.DataFrame([[1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, 1],
                                      [4, 4, 4]])),
    ]
)
def test_hold(simple_factor, period, expected):
    simple_factor.hold(period)
    assert_allclose(simple_factor.data, expected, equal_nan=True)


@pytest.mark.parametrize('mask, expected',
    [
        pytest.param(pd.DataFrame([[True, False, True],
                                   [False, True, False],
                                   [True, True, True],
                                   [False, False, False]]),
                     pd.DataFrame([[1, np.nan, 1],
                                   [np.nan, 2, np.nan],
                                   [3, 3, 3],
                                   [np.nan, np.nan, np.nan]]),
        ),
        pytest.param(pd.DataFrame([[True, False, True],
                                   [False, True, False],
                                   [True, True, True],
                                   [False, False, False],
                                   [False, True, False]]),
                    pd.DataFrame([[1, np.nan, 1],
                                  [np.nan, 2, np.nan],
                                  [3, 3, 3],
                                  [np.nan, np.nan, np.nan]])),
        pytest.param(pd.DataFrame([[False, True, False],
                                   [True, True, True],
                                   [False, False, False]], index=[1, 2, 3]),
                     pd.DataFrame([[np.nan, 2, np.nan],
                                   [3, 3, 3],
                                   [np.nan, np.nan, np.nan]])),
    ]
)
def test_filter(simple_factor, mask, expected):
    simple_factor.filter(mask)
    assert_allclose(simple_factor.data, expected, equal_nan=True)
