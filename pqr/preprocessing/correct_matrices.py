from typing import List

import pandas as pd


def correct_matrices(*matrices: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Function for correcting matrices: bringing them to a single view with
    similar indices and columns in the same order.

    Parameters
    ----------
    matrices : iterable of pd.DataFrame
        Matrices to be corrected.

    Returns
    -------
    list of pd.DataFrame
        Corrected matrices. It is guaranteed that all matrices have the same
        indices and columns in the same order.
    """

    # collect all
    all_columns = set()
    all_indices = set()
    for matrix in matrices:
        all_columns |= set(matrix.columns)
        all_indices |= set(matrix.index)

    # find what to drop
    columns_to_drop = set()
    indices_to_drop = set()
    for matrix in matrices:
        columns_to_drop |= (all_columns - set(matrix.columns))
        indices_to_drop |= (all_indices - set(matrix.index))

    # drop and sort
    corrected_matrices = []
    for matrix in matrices:
        # drop and sort columns
        matrix_corrected = matrix.drop(columns_to_drop, axis=1,
                                       errors='ignore')
        matrix_corrected = matrix_corrected[
            sorted(matrix_corrected.columns.values)]
        # drop and sort indices
        matrix_corrected = matrix_corrected.drop(indices_to_drop, axis=0,
                                                 errors='ignore')
        matrix_corrected = matrix_corrected.sort_index()

        corrected_matrices.append(matrix_corrected)

    return corrected_matrices
