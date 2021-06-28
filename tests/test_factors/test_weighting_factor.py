import pytest
import numpy as np
import pandas as pd

from pqr.factors import WeightingFactor, WeightingMultiFactor


@pytest.mark.parametrize(
    'factor, prices, answer',
    (
        # test 1: static weighting factor
        [
            WeightingFactor(pd.DataFrame([[1, 2, 3, 4],
                                          [2, 3, 4, 5],
                                          [5, 5, 7, 8],
                                          [10, 20, 31, -1]]), False),
            pd.DataFrame([[0, 0, 0, 0],
                          [1, 0, 1, 0],
                          [1, 1, 1, 1],
                          [0, 1, 1, 1]]),
            pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                          [1/4, 0, 3/4, 0],
                          [2/14, 3/14, 4/14, 5/14],
                          [0, 5/20, 7/20, 8/20]]),
        ],
        # test 2: dynamic weighting factor
        [
            WeightingFactor(pd.DataFrame([[1, 2, 3, 4],
                                          [2, 3, 4, 5],
                                          [5, 5, 7, 8],
                                          [10, 20, 31, -1]]), True),
            pd.DataFrame([[0, 0, 0, 0],
                          [1, 0, 1, 0],
                          [1, 1, 1, 1],
                          [0, 1, 1, 1]]),
            pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [1 / 2.083, 0.5 / 2.083, 0.33 / 2.083, 0.33 / 2.083],
                          [0, 0.66 / 2.016, 0.75 / 2.016, 0.6 / 2.016]]),
        ],
    )
)
def test_weighting_factor(
        factor,
        prices,
        answer
):
    assert np.allclose(
        np.nan_to_num(factor.weigh(prices)), np.nan_to_num(answer),
        atol=0.1, equal_nan=True
    )
