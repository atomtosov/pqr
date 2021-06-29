import numpy as np
import pandas as pd
import pytest

from pqr.factors import PickingFactor, WeighMultiFactor


@pytest.mark.parametrize(
    'data, dynamic, bigger_better, '
    'looking_period, lag_period, answer',
    (
        # test 1: static single factor, looking=1, lag=0
        [
            pd.DataFrame([[1, 2, 3, 4],
                          [2, 3, 4, 5],
                          [5, 5, 7, 8],
                          [10, 20, 31, -1]]),
            False,
            True,
            1,
            0,
            pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                          [1, 2, 3, 4],
                          [2, 3, 4, 5],
                          [5, 5, 7, 8]])
        ],
        # test 2: static single factor, looking=1, lag=1
        [
            pd.DataFrame([[1, 2, 3, 4],
                          [2, 3, 4, 5],
                          [5, 5, 7, 8],
                          [10, 20, 31, -1]]),
            False,
            True,
            1,
            1,
            pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [1, 2, 3, 4],
                          [2, 3, 4, 5]])
        ],
        # test 3: static single factor, looking=2, lag=1
        [
            pd.DataFrame([[1, 2, 3, 4],
                          [2, 3, 4, 5],
                          [5, 5, 7, 8],
                          [10, 20, 31, -1]]),
            False,
            True,
            2,
            1,
            pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [1, 2, 3, 4]])
        ],
        # test 4: dynamic single factor, looking=1, lag=0
        [
            pd.DataFrame([[1, 2, 3, 4],
                          [2, 3, 4, 5],
                          [5, 5, 7, 8],
                          [10, 20, 31, -1]]),
            True,
            True,
            1,
            0,
            pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [2/1-1, 3/2-1, 4/3-1, 5/4-1],
                          [5/2-1, 5/3-1, 7/4-1, 8/5-1]])
        ],
        # test 5: dynamic single factor, looking=1, lag=1
        [
            pd.DataFrame([[1, 2, 3, 4],
                          [2, 3, 4, 5],
                          [5, 5, 7, 8],
                          [10, 20, 31, -1]]),
            True,
            True,
            1,
            1,
            pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [2/1-1, 3/2-1, 4/3-1, 5/4-1]])
        ],
        # test 6: dynamic single factor, looking=2, lag=0
        [
            pd.DataFrame([[1, 2, 3, 4],
                          [2, 3, 4, 5],
                          [5, 5, 7, 8],
                          [10, 20, 31, -1]]),
            True,
            True,
            2,
            0,
            pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan],
                          [5/1-1, 5/2-1, 7/3-1, 8/4-1]])
        ],
    )
)
def test_single_factor_transform(
        data,
        dynamic,
        bigger_better,
        looking_period,
        lag_period,
        answer
):
    assert np.all(
        np.nan_to_num(
            PickingFactor(data, dynamic).transform(looking_period, lag_period)
        ) == np.nan_to_num(answer)
    )


@pytest.mark.parametrize(
    'factors, looking_period, lag_period, answer',
    (
        # test 1: 2 static factors, looking=1, lag=0
        [
            [
                PickingFactor(pd.DataFrame([[1, 2, 3, 4],
                                            [2, 3, 4, 5],
                                            [5, 5, 7, 8],
                                            [10, 20, 31, -1]]), False),
                PickingFactor(pd.DataFrame([[2, 3, 4, 5],
                                            [6, 17, 25, 41],
                                            [8, 19, -1, 5],
                                            [5, 5, 7, 8]]), False),
            ],
            1,
            0,
            (
                    pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                                  [1, 2, 3, 4],
                                  [2, 3, 4, 5],
                                  [5, 5, 7, 8]]),
                    pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                                  [2, 3, 4, 5],
                                  [6, 17, 25, 41],
                                  [8, 19, -1, 5]])
            )
        ],
        # test 2: 2 dynamic factors, looking=2, lag=0
        [
            [
                PickingFactor(pd.DataFrame([[1, 2, 3, 4],
                                            [2, 3, 4, 5],
                                            [5, 5, 7, 8],
                                            [10, 20, 31, -1]]), True),
                PickingFactor(pd.DataFrame([[2, 3, 4, 5],
                                            [6, 17, 25, 41],
                                            [8, 19, -1, 5],
                                            [5, 5, 7, 8]]), True),
            ],
            2,
            0,
            (
                    pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan],
                                  [5/1-1, 5/2-1, 7/3-1, 8/4-1]]),
                    pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan],
                                  [8/2-1, 19/3-1, -1/4-1, 5/5-1]])
            )
        ],
        # test 3: dynamic & static factors, looking=2, lag=0
        [
            [
                PickingFactor(pd.DataFrame([[1, 2, 3, 4],
                                            [2, 3, 4, 5],
                                            [5, 5, 7, 8],
                                            [10, 20, 31, -1]]), False),
                PickingFactor(pd.DataFrame([[2, 3, 4, 5],
                                            [6, 17, 25, 41],
                                            [8, 19, -1, 5],
                                            [5, 5, 7, 8]]), True),
            ],
            2,
            0,
            (
                    pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan],
                                  [1, 2, 3, 4],
                                  [2, 3, 4, 5]]),
                    pd.DataFrame([[np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan],
                                  [8/2-1, 19/3-1, -1/4-1, 5/5-1]])
            )
        ],
    )
)
def test_multi_factor_transform(
        factors,
        looking_period,
        lag_period,
        answer
):
    assert np.all(
        np.nan_to_num(
            WeighMultiFactor(factors).transform(looking_period, lag_period)
        ) == np.nan_to_num(answer)
    )
