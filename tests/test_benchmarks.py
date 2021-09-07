import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pqr import Benchmark


@pytest.mark.parametrize(
    ['index_values', 'expected'],
    [
        # test 1
        pytest.param(
            pd.Series([1, 2, 3, 4, 5]),
            pd.Series([np.nan, 1, 0.5, 1/3, 0.25]),
            id='only increasing'
        ),
        # test 2
        pytest.param(
            pd.Series([5, 4, 3, 2, 1]),
            pd.Series([np.nan, -0.2, -0.25, -1/3, -0.5]),
            id='only decreasing'
        ),
        # test 3
        pytest.param(
            pd.Series([1, 2, 1, 3, 2]),
            pd.Series([np.nan, 1, -0.5, 2, -1/3]),
            id='mixed'
        ),
    ]
)
def test_from_index(
        index_values: pd.Series,
        expected: pd.Series
):
    benchmark = Benchmark().from_index(index_values)
    assert_allclose(benchmark.returns, expected)
