import pytest
import numpy as np

from pqr.benchmarks import CustomBenchmark


@pytest.mark.parametrize(
    'prices, weighting_factor, answer',
    (
    )
)
def test_benchmark_returns(
        prices,
        weighting_factor,
        answer
):
    assert np.all(
        CustomBenchmark(prices, weighting_factor).returns == answer
    )
