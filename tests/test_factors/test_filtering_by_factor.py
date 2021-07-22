import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pqr.factors import Factor, Filter
from pqr.thresholds import Thresholds

test_data = [
    # Single Factors
    # test 1: static single factor min_threshold=3
    pytest.param(
        Factor(pd.DataFrame([[1, 1, 1, 1],
                             [2, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4],
                             [5, 5, 5, 5]]),
               dynamic=False),
        Thresholds(lower=3),
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]]),
        pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]]),
        id='static single filtering min_threshold=3'
    ),
    # test 2: static single factor max_threshold=3 + corrupted prices
    pytest.param(
        Factor(pd.DataFrame([[1, 1, 1, 1],
                             [2, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4],
                             [5, 5, 5, 5]]),
               dynamic=False),
        Thresholds(upper=3),
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, np.nan, np.nan, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]]),
        pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                      [1, 1, 1, 1],
                      [1, np.nan, np.nan, 1],
                      [1, 1, 1, 1],
                      [np.nan, np.nan, np.nan, np.nan]]),
        id='static single filtering max_threshold=3 + corrupted prices'
    ),
    # test 3: dynamic single factor min_threshold=1/4, max_threshold=1/2 + c.p.
    pytest.param(
        Factor(pd.DataFrame([[1, 1, 1, 1],
                             [2, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4],
                             [5, 5, 5, 5]]),
               dynamic=True),
        Thresholds(1 / 4, 1 / 2),
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [np.nan, 1, 1, 1],
                      [1, 1, 1, np.nan]]),
        pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [np.nan, 1, 1, 1],
                      [1, 1, 1, np.nan]]),
        id='dynamic single filtering min_threshold=1/4, max_threshold=1/2 '
           '+ corrupted prices'
    ),
]


@pytest.mark.parametrize(
    'factor, thresholds, prices, expected',
    test_data
)
def test_filtering_by_factor(
        factor: Factor,
        thresholds: Thresholds,
        prices: pd.DataFrame,
        expected: pd.DataFrame
):
    assert_allclose(
        Filter(factor, thresholds)(prices),
        expected,
        equal_nan=True
    )
