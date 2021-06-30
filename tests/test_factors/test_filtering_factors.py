import pytest
from numpy.testing import assert_allclose

import numpy as np
import pandas as pd

from pqr.factors import FilteringFactor, FilteringMultiFactor

test_data = [
    # Single Factors
    # test 1: static single factor min_threshold=3
    pytest.param(
        FilteringFactor(pd.DataFrame([[1, 1, 1, 1],
                                      [2, 2, 2, 2],
                                      [3, 3, 3, 3],
                                      [4, 4, 4, 4],
                                      [5, 5, 5, 5]]),
                        dynamic=False,
                        min_threshold=3),
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
        FilteringFactor(pd.DataFrame([[1, 1, 1, 1],
                                      [2, 2, 2, 2],
                                      [3, 3, 3, 3],
                                      [4, 4, 4, 4],
                                      [5, 5, 5, 5]]),
                        dynamic=False,
                        max_threshold=3),
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
        FilteringFactor(pd.DataFrame([[1, 1, 1, 1],
                                      [2, 2, 2, 2],
                                      [3, 3, 3, 3],
                                      [4, 4, 4, 4],
                                      [5, 5, 5, 5]]),
                        dynamic=True,
                        min_threshold=1/4,
                        max_threshold=1/2),
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
    # Multi Factors
    # test 1: 2 static factors min_threshold=(None, 2), max_threshold=(3, None)
    pytest.param(
        FilteringMultiFactor(
            [
                FilteringFactor(pd.DataFrame([[1, 1, 1, 1],
                                              [2, 2, 2, 2],
                                              [3, 3, 3, 3],
                                              [4, 4, 4, 4],
                                              [5, 5, 5, 5]]),
                                dynamic=False,
                                max_threshold=3),
                FilteringFactor(pd.DataFrame([[1, 1, 1, 1],
                                              [2, 2, 2, 2],
                                              [3, 3, 3, 3],
                                              [4, 4, 4, 4],
                                              [5, 5, 5, 5]]),
                                dynamic=False,
                                min_threshold=2),
            ]
        ),
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]]),
        pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [np.nan, np.nan, np.nan, np.nan]]),
        id='2 static multi min_threshold=(None, 2), max_threshold=(3, None)'
    ),
    # test 2: 2 different factors
    # min_threshold=(None, 1), max_threshold=(3, None) + c.p.
    pytest.param(
        FilteringMultiFactor(
            [
                FilteringFactor(pd.DataFrame([[1, 1, 1, 1],
                                              [2, 2, 2, 2],
                                              [3, 3, 3, 3],
                                              [4, 4, 4, 4],
                                              [5, 5, 5, 5]]),
                                dynamic=False,
                                max_threshold=3),
                FilteringFactor(pd.DataFrame([[1, 1, 1, 1],
                                              [2, 2, 2, 2],
                                              [3, 3, 3, 3],
                                              [4, 4, 4, 4],
                                              [5, 5, 5, 5]]),
                                dynamic=True,
                                min_threshold=1/2),
            ]
        ),
        pd.DataFrame([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, np.nan],
                      [np.nan, 1, 1, 1],
                      [1, 1, 1, 1]]),
        pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [1, 1, 1, np.nan],
                      [np.nan, 1, 1, 1],
                      [np.nan, np.nan, np.nan, np.nan]]),
        id='2 static multi min_threshold=(None, 2), max_threshold=(3, None)'
    )
]


@pytest.mark.parametrize(
    'factor, prices, expected',
    test_data
)
def test_filtering_factor(
        factor,
        prices,
        expected
):
    assert_allclose(
        factor.filter(prices),
        expected,
        equal_nan=True
    )
