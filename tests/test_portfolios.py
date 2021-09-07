import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pqr import Factor, Portfolio


@pytest.mark.parametrize(
    ['data', 'thresholds', 'better', 'expected'],
    [
        # test 1
        pytest.param(
            pd.DataFrame([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [10, 11, 12]]),
            (0, 1/3), 'less',
            pd.DataFrame([[1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0]]),
            id='q(0, 1/3)'
        ),
        # test 2
        pytest.param(
            pd.DataFrame([[np.nan, 2, 3],
                          [4, np.nan, 6],
                          [7, 8, 9],
                          [np.nan, np.nan, 12]]),
            (0, 1/2), 'less',
            pd.DataFrame([[0, 1, 0],
                          [1, 0, 0],
                          [1, 1, 0],
                          [0, 0, 1]]),
            id='q(0, 1/2) with nans'
        ),
    ]
)
def test_picking_by_factor_quantile(data, thresholds, better, expected):
    portfolio = Portfolio()
    factor = Factor(data)
    portfolio.pick_by_factor(factor, thresholds, better, method='quantile')
    assert_allclose(portfolio.picks, expected)


@pytest.mark.parametrize(
    ['data', 'thresholds', 'better', 'expected'],
    [
        # test 1
        pytest.param(
            pd.DataFrame([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [10, 11, 12]]),
            (1, 2), 'less',
            pd.DataFrame([[1, 1, 0],
                          [1, 1, 0],
                          [1, 1, 0],
                          [1, 1, 0]]),
            id='top(1, 2)'
        ),
        # test 2
        pytest.param(
            pd.DataFrame([[np.nan, 2, 3],
                          [4, np.nan, 6],
                          [7, 8, 9],
                          [np.nan, np.nan, 12]]),
            (1, 2), 'less',
            pd.DataFrame([[0, 1, 1],
                          [1, 0, 1],
                          [1, 1, 0],
                          [0, 0, 1]]),
            id='top(1, 2) with nans'
        ),
    ]
)
def test_picking_by_factor_top(data, thresholds, better, expected):
    portfolio = Portfolio()
    factor = Factor(data)
    portfolio.pick_by_factor(factor, thresholds, better, method='top')
    assert_allclose(portfolio.picks, expected)


@pytest.mark.parametrize(
    ['data', 'thresholds', 'expected'],
    [
        # test 1
        pytest.param(
            pd.DataFrame([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [10, 11, 12]]),
            (2, 8),
            pd.DataFrame([[0, 1, 1],
                          [1, 1, 1],
                          [1, 1, 0],
                          [0, 0, 0]]),
            id='time-series(2, 8)'
        ),
        # test 2
        pytest.param(
            pd.DataFrame([[np.nan, 2, 3],
                          [4, np.nan, 6],
                          [7, 8, 9],
                          [np.nan, np.nan, 12]]),
            (2, 8),
            pd.DataFrame([[0, 1, 1],
                          [1, 0, 1],
                          [1, 1, 0],
                          [0, 0, 0]]),
            id='time-series(2, 8) with nans'
        ),
    ]
)
def test_picking_by_factor_time_series(data, thresholds, expected):
    portfolio = Portfolio()
    factor = Factor(data)
    portfolio.pick_by_factor(factor, thresholds, method='time-series')
    assert_allclose(portfolio.picks, expected)


@pytest.mark.parametrize(
    ['data', 'picks', 'expected'],
    [
        # test 1
        pytest.param(
            pd.DataFrame([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [10, 11, 12]]),
            pd.DataFrame([[True, False, False],
                          [True, False, False],
                          [True, False, False],
                          [True, False, False]]),
            pd.DataFrame([[1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0]]),
            id='all weights are equal to 1'
        ),
        # test 2
        pytest.param(
            pd.DataFrame([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [10, 11, 12]]),
            pd.DataFrame([[True, True, False],
                          [True, True, False],
                          [True, True, False],
                          [True, True, False]]),
            pd.DataFrame([[1 / 3, 2 / 3, 0],
                          [4 / 9, 5 / 9, 0],
                          [7 / 15, 8 / 15, 0],
                          [10 / 21, 11 / 21, 0]]),
            id='different weights'
        ),
        # test 3
        pytest.param(
            pd.DataFrame([[np.nan, 2, 3],
                          [4, np.nan, 6],
                          [7, 8, 9],
                          [np.nan, np.nan, 12]]),
            pd.DataFrame([[True, True, False],
                          [True, True, False],
                          [True, True, False],
                          [True, True, False]]),
            pd.DataFrame([[0, 1, 0],
                          [1, 0, 0],
                          [7 / 15, 8 / 15, 0],
                          [0, 0, 0]]),
            id='different weights with nans'
        ),
    ]
)
def test_weighing_by_factor(data, picks, expected):
    portfolio = Portfolio()
    portfolio.picks = picks
    factor = Factor(data)
    portfolio.weigh_by_factor(factor)
    assert_allclose(portfolio.weights, expected)


@pytest.mark.parametrize(
    ['data', 'target', 'weights', 'expected'],
    [
        # test 1
        pytest.param(
            pd.DataFrame([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [10, 11, 12]]),
            10,
            pd.DataFrame([[1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0]]),
            pd.DataFrame([[0.1, 0, 0],
                          [0.4, 0, 0],
                          [0.7, 0, 0],
                          [1, 0, 0]]),
            id='simple scaling'
        ),
        # test 2
        pytest.param(
            pd.DataFrame([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [10, 11, 12]]),
            5,
            pd.DataFrame([[1 / 3, 2 / 3, 0],
                          [4 / 9, 5 / 9, 0],
                          [7 / 15, 8 / 15, 0],
                          [10 / 21, 11 / 21, 0]]),
            pd.DataFrame([[1 / 15, 4 / 15, 0],
                          [16 / 45, 5 / 9, 0],
                          [49 / 75, 64 / 75, 0],
                          [20 / 21, 121 / 105, 0]]),
            id='different scales'
        ),
        # test 3
        pytest.param(
            pd.DataFrame([[np.nan, 2, 3],
                          [4, np.nan, 6],
                          [7, 8, 9],
                          [np.nan, np.nan, 12]]),
            10,
            pd.DataFrame([[1, 0, 0],
                          [1, 0, 0],
                          [7 / 15, 8 / 15, 0],
                          [0, 0, 0]]),
            pd.DataFrame([[0, 0, 0],
                          [0.4, 0, 0],
                          [49 / 150, 64 / 150, 0],
                          [0, 0, 0]]),
            id='different scales with nans'
        ),
    ]
)
def test_scaling_by_factor(data, weights, target, expected):
    portfolio = Portfolio()
    portfolio.weights = weights
    factor = Factor(data)
    portfolio.scale_by_factor(factor, target)
    assert_allclose(portfolio.weights, expected)
