import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pqr import Factor


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
def test_prefilter(
        data: pd.DataFrame,
        mask: pd.DataFrame,
        expected: pd.DataFrame
):
    factor = Factor(data)
    factor.prefilter(mask)
    assert_allclose(factor.data, expected, equal_nan=True)
