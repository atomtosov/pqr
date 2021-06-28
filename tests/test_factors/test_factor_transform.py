import numpy as np
import pytest

from pqr.factors import Factor


@pytest.mark.parametrize(
    'factor, data, dynamic, bigger_better, '
    'looking_period, lag_period, answer'
)
def test_factor_transform(
        factor,
        data,
        dynamic,
        bigger_better,
        looking_period,
        lag_period,
        answer
):
    assert np.all(
        np.nan_to_num(
            factor(data, dynamic).transform(looking_period, lag_period)
        ) == np.nan_to_num(answer)
    )
