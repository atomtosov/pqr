import pandas as pd

from ..interfaces import IFiltering


class NoFilter(IFiltering):
    """
    Class for dummy-filtering.
    """

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Method to do not filter stock universe.

        Parameters
        ----------
        data : pd.DataFrame
            Data to not be filtered at all.

        Returns
        -------
        pd.DataFrame
            The same dataframe as passed.
        """

        return data
