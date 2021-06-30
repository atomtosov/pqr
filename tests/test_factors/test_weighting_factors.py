import pytest
from numpy.testing import assert_allclose

import numpy as np
import pandas as pd

from pqr.factors import WeightingFactor, WeightingMultiFactor

test_data = [
    # Single Factors
    # test 1: static single factor
    pytest.param(
        WeightingFactor(pd.DataFrame([[1, 2, 3, 4],
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
        pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                      [1/4, 0, 3/4, 0],
                      [1, 0, 0, 0],
                      [0, 2/9, 3/9, 4/9],
                      [1/10, 2/10, 3/10, 4/10]]),
        id='static single weighting'
    ),
    # test 2: dynamic single factor
    pytest.param(
        WeightingFactor(pd.DataFrame([[1, 2, 3, 4],
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
        pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [1, 0, 0, 0],
                      [0, 2/9, 3/9, 4/9],
                      [1/10, 2/10, 3/10, 4/10]]),
        id='dynamic single weighting'
    ),
    # test 3: 2 dynamic multi weighting
    pytest.param(
        WeightingMultiFactor(
            [
                WeightingFactor(pd.DataFrame([[1, 2, 3, 4],
                                              [2, 6, 12, 20],
                                              [4, 18, 48, 100],
                                              [8, 54, 192, 500],
                                              [16, 162, 768, 2500]]),
                                dynamic=True),
                WeightingFactor(pd.DataFrame([[1, 2, 3, 4],
                                              [2, 6, 12, 20],
                                              [4, 18, 48, 100],
                                              [8, 54, 192, 500],
                                              [16, 162, 768, 2500]]),
                                dynamic=True),
            ]
        ),
        pd.DataFrame([[0, 0, 0, 0],
                      [1, 0, 1, 0],
                      [1, 0, 0, 0],
                      [0, 1, 1, 1],
                      [1, 1, 1, 1]]),
        pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan],
                      [1, 0, 0, 0],
                      [0, 4/29, 9/29, 16/29],
                      [1/30, 4/30, 9/30, 16/30]]),
        id='2 dynamic multi weighting'
    ),
]


@pytest.mark.parametrize(
    'factor, positions, expected',
    test_data
)
def test_weighting_factor(
        factor,
        positions,
        expected
):
    assert_allclose(
        factor.weigh(positions),
        expected,
        equal_nan=True
    )
