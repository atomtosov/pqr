from typing import Iterable, Any, List

import numpy as np
import pandas as pd


def replace_with_nan(*matrices: pd.DataFrame,
                     to_replace: Iterable[Any]) -> List[pd.DataFrame]:
    replaced_matrices = []
    for matrix in matrices:
        replaced_matrix = matrix.replace(to_replace, np.nan)
        replaced_matrices.append(replaced_matrix)
    return replaced_matrices
