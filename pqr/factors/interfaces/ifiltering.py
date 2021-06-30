from abc import abstractmethod

import pandas as pd


class IFiltering:
    """
    Interface for filtering factors.
    """

    @abstractmethod
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Method for filtering stock universe by factor values and some
        restrictions on it.
        """
