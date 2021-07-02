from enum import Enum

import pandas as pd


class DataPeriodicityToPandas(Enum):
    yearly = 'A'
    quarterly = 'Q'
    monthly = 'M'
    weekly = 'W'
    daily = 'D'


class DataPeriodicity(Enum):
    # yearly
    A = 1
    # quarterly
    Q = 4
    # monthly
    M = 12
    # weekly
    W = 52
    # daily
    D = 252


def resample(*matrices: pd.DataFrame, periodicity='monthly'):
    resampled_matrices = []
    for matrix in matrices:
        matrix = matrix.resample(
            DataPeriodicityToPandas[periodicity].value
        ).asfreq()
        resampled_matrices.append(matrix)
    return resampled_matrices
