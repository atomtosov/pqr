import pytest
import numpy as np

from pqr.factors import FilteringFactor, FilteringMultiFactor


@pytest.mark.parametrize(
    'factor, data, dynamic, bigger_better, '
    'prices, answer',
    (
        ...
    )
)
def test_filtering_factor(
        factor,
        data,
        dynamic,
        bigger_better,
        prices,
        answer
):
    assert np.all(
        factor(data, dynamic, bigger_better).
        filter(prices) == answer
    )
