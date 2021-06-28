import pytest
import numpy as np
import pandas as pd

from pqr.factors import FilteringFactor, FilteringMultiFactor


@pytest.mark.parametrize(
    'factor, prices, answer',
    (
        # test 1: static filtering factor(7, inf)
        [
            FilteringFactor(pd.DataFrame([[1, 2, 3, 4],
                                          [2, 3, 4, 5],
                                          [5, 5, 7, 8],
                                          [10, 20, 31, -1]]), False,
                            min_threshold=7),
            pd.DataFrame([[1, 2, np.nan, 4],
                          [2, np.nan, np.nan, 5],
                          [4, 5, np.nan, 8],
                          [8, 20, np.nan, -1]]),
            pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, -1]]),
        ],
        # test 2: dynamic filtering factor(1, 4)
        [
            FilteringFactor(pd.DataFrame([[1, 2, 3, 4],
                                          [2, 3, 4, 5],
                                          [5, 5, 7, 8],
                                          [10, 20, 31, -1]]), True,
                            min_threshold=1,
                            max_threshold=4),
            pd.DataFrame([[1, 2, np.nan, 4],
                          [2, np.nan, np.nan, 5],
                          [4, 5, np.nan, 8],
                          [8, 20, np.nan, -1]]),
            pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [4, np.nan, np.nan, np.nan],
                          [8, np.nan, np.nan, np.nan]]),
        ],
    )
)
def test_filtering_factor(
        factor,
        prices,
        answer
):
    assert np.all(
        np.nan_to_num(factor.filter(prices)) == np.nan_to_num(answer)
    )
