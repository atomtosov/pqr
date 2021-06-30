from abc import abstractmethod

import pandas as pd


class IWeighting:
    """
    Interface for weighting factors.
    """

    @abstractmethod
    def weigh(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Method to weigh data (e.g. positions) by factor values.
        """
