from enum import Enum

import pandas as pd


class DataPeriodicity(Enum):
    yearly = 'A'
    quarterly = 'Q'
    monthly = 'M'
    weekly = 'W'
    daily = 'D'


def resample(*matrices: pd.DataFrame, periodicity='monthly'):
    resampled_matrices = []
    for matrix in matrices:
        matrix = matrix.resample(DataPeriodicity[periodicity].value).asfreq()
        resampled_matrices.append(matrix)
    return resampled_matrices
