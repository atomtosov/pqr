from typing import List

import pandas as pd


_DATA_PERIODICITY_TO_PANDAS_ALIASES = {
    'yearly': 'A',
    'quarterly': 'Q',
    'monthly': 'M',
    'weekly': 'W',
    'daily': 'B'
}


def resample(*matrices: pd.DataFrame,
             periodicity: str = 'monthly') -> List[pd.DataFrame]:
    """
    Function for resampling data.

    Parameters
    ----------
    matrices : iterable of pd.DataFrame
        Matrices to be resampled.
    periodicity : str
        Periodicity of data to set it into index of every matrix.

    Returns
    -------
    list of pd.DataFrame
        Resampled matrices. It is guaranteed that all of them have the same
        periodicity.
    """

    resampled_matrices = []
    for matrix in matrices:
        matrix = matrix.resample(
            _DATA_PERIODICITY_TO_PANDAS_ALIASES[periodicity]
        ).asfreq()
        resampled_matrices.append(matrix)
    return resampled_matrices
