import pytest
from numpy.testing import assert_allclose

import numpy as np
import pandas as pd

from pqr import Benchmark


@pytest.fixture
def benchmark():
    return Benchmark()


@pytest.mark.parametrize('index_values, expected',
    [
        pytest.param(pd.Series([1, 2, 3, 4, 5]), 
                     pd.Series([np.nan, 1, 0.5, 1/3, 0.25])),
        pytest.param(pd.Series([5, 4, 3, 2, 1]), 
                     pd.Series([np.nan, -0.2, -0.25, -1/3, -0.5])),
        pytest.param(pd.Series([1, 2, 1, 3, 2]),
                     pd.Series([np.nan, 1, -0.5, 2, -1/3])),
    ]
)
def test_from_index(benchmark, index_values, expected):
    benchmark.from_index(index_values)
    assert_allclose(benchmark.returns, expected)
