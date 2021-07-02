import pandas as pd


_DATA_PERIODICITY_TO_PANDAS_ALIASES = {
    'yearly': 'A',
    'quarterly': 'Q',
    'monthly': 'M',
    'weekly': 'W',
    'daily': 'B'
}


def resample(*matrices: pd.DataFrame, periodicity='monthly'):
    resampled_matrices = []
    for matrix in matrices:
        matrix = matrix.resample(
            _DATA_PERIODICITY_TO_PANDAS_ALIASES[periodicity]
        ).asfreq()
        resampled_matrices.append(matrix)
    return resampled_matrices
