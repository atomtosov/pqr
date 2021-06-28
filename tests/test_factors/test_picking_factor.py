import pytest
import numpy as np

from pqr.factors import Factor
from pqr.intervals import Quantiles


@pytest.mark.parametrize(
    'factor, data, dynamic, bigger_better, '
    'prices, interval, looking_period, lag_period, answer',
    (
        ...
    )
)
def test_picking_factor(
        factor,
        data,
        dynamic,
        bigger_better,
        prices,
        interval,
        looking_period,
        lag_period,
        answer
):
    assert np.all(
        factor(data, dynamic, bigger_better).
        pick(prices, interval, looking_period, lag_period) == answer
    )
