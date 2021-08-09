import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pqr import Factor, Portfolio


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
        )
    ]
)
def test_weighing_by_factor(
        data: pd.DataFrame,
        picks: pd.DataFrame,
        expected: pd.DataFrame
):
    portfolio = Portfolio()
    portfolio.picks = picks
    factor = Factor(data)
    portfolio.weigh_by_factor(factor)
    assert_allclose(portfolio.weights, expected)
