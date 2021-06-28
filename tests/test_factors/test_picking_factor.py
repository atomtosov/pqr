import pytest
import numpy as np
import pandas as pd

from pqr.factors import Factor
from pqr.intervals import Quantiles


@pytest.mark.parametrize(
    'factor, prices, interval, looking_period, lag_period, answer',
    (
        # test 1: static single factor(1, 0), q(0, 1/3)
        [
            Factor(pd.DataFrame([[1, 2, 3, 4],
                                 [2, 3, 4, 5],
                                 [5, 5, 7, 8],
                                 [10, 20, 31, -1]]), False),
            pd.DataFrame([[1, 2, np.nan, 4],
                          [2, np.nan, np.nan, 5],
                          [4, 5, np.nan, 8],
                          [8, 20, np.nan, -1]]),
            Quantiles(0, 1/3),
            1,
            0,
            pd.DataFrame([[False, False, False, False],
                          [True, False, False, False],
                          [True, False, False, False],
                          [True, True, False, False]])
        ],
        # test 2: dynamic single factor(2, 0), q(0, 1/3)
[
            Factor(pd.DataFrame([[1, 2, 3, 4],
                                 [2, 3, 4, 5],
                                 [5, 5, 7, 8],
                                 [10, 20, 31, -1]]), True),
            pd.DataFrame([[1, 2, np.nan, 4],
                          [2, np.nan, np.nan, 5],
                          [4, 5, np.nan, 8],
                          [8, 20, np.nan, -1]]),
            Quantiles(0, 1/3),
            2,
            0,
            pd.DataFrame([[False, False, False, False],
                          [False, False, False, False],
                          [False, False, False, False],
                          [False, False, False, True]])
        ],
    )
)
def test_picking_factor(
        factor,
        prices,
        interval,
        looking_period,
        lag_period,
        answer
):
    assert np.all(
        np.nan_to_num(
            factor.pick(prices, interval, looking_period, lag_period)
        ) == np.nan_to_num(answer)
    )
