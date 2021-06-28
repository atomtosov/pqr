import pytest
import numpy as np

from pqr.benchmarks import Benchmark


@pytest.mark.parametrize(
    'prices, answer',
    (
        ...
    )
)
def test_benchmark_returns(
        prices,
        answer
):
    assert np.all(
        Benchmark(prices).returns == answer
    )
