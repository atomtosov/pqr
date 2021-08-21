import pandas as pd


def align(*dataframes):
    common_index = dataframes[0].index
    common_columns = slice(None)

    for dataframe in dataframes:
        if isinstance(dataframe, pd.DataFrame):
            common_index = common_index.intersection(dataframe.index)
            try:
                common_columns = common_columns.intersection(dataframe.columns)
            except AttributeError:
                common_columns = dataframe.columns

        elif isinstance(dataframe, pd.Series):
            common_index = common_index.intersection(dataframe.index)

    dataframes = list(dataframes)
    for i, dataframe in enumerate(dataframes):
        if isinstance(dataframe, pd.DataFrame):
            dataframes[i] = dataframe.loc[common_index, common_columns]
        elif isinstance(dataframe, pd.Series):
            dataframes[i] = dataframe.loc[common_index]

    return dataframes


def get_annualization_factor(dataframe):
    freq_alias_to_num = {
        'BA': 1, 'A': 1,     # yearly
        'BQ': 4, 'Q': 4,     # quarterly
        'BM': 12, 'M': 12,   # monthly
        'W': 52,             # weekly
        'B': 252, 'D': 252,  # daily
    }

    inferred_freq = getattr(dataframe.index, 'inferred_freq', None)
    freq_num = freq_alias_to_num.get(inferred_freq)
    if freq_num is None:
        raise ValueError('periodicity of given dataframe cannot be defined, '
                         'try to resample data')
    return freq_num
