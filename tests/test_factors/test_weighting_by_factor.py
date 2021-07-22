import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pqr.factors import Factor, Weigher

test_data = [
    # test 1: static single factor
    pytest.param(
        Factor(pd.DataFrame([[1, 2, 3, 4],
                             [1, 2, 3, 4],
                             [1, 2, 3, 4],
                             [1, 2, 3, 4],
                             [1, 2, 3, 4]]),
               dynamic=False),
        pd.DataFrame([[0, 0, 0, 0],
                      [1, 0, 1, 0],
                      [1, 0, 0, 0],
                      [0, 1, 1, 1],
                      [1, 1, 1, 1]]),
        pd.DataFrame([[0, 0, 0, 0],
                      [1 / 4, 0, 3 / 4, 0],
                      [1, 0, 0, 0],
                      [0, 2 / 9, 3 / 9, 4 / 9],
                      [1 / 10, 2 / 10, 3 / 10, 4 / 10]]),
        id='static single weighting'
    ),
    # test 2: dynamic single factor
    pytest.param(
        Factor(pd.DataFrame([[1, 2, 3, 4],
                             [2, 6, 12, 20],
                             [4, 18, 48, 100],
                             [8, 54, 192, 500],
                             [16, 162, 768, 2500]]),
               dynamic=True),
        pd.DataFrame([[0, 0, 0, 0],
                      [1, 0, 1, 0],
                      [1, 0, 0, 0],
                      [0, 1, 1, 1],
                      [1, 1, 1, 1]]),
        pd.DataFrame([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 2 / 9, 3 / 9, 4 / 9],
                      [1 / 10, 2 / 10, 3 / 10, 4 / 10]]),
        id='dynamic single weighting'
    ),
]


@pytest.mark.parametrize(
    'factor, positions, expected',
    test_data
)
def test_weighting_by_factor(
        factor: Factor,
        positions: pd.DataFrame,
        expected: pd.DataFrame
):
    assert_allclose(
        Weigher(factor)(positions),
        expected,
        equal_nan=True
    )
