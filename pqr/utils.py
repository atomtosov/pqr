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
        'A': 1, 'AS': 1, 'BYS': 1, 'BA': 1, 'BAS': 1, 'RE': 1,          # yearly
        'Q': 4, 'QS': 4, 'BQ': 4, 'BQS': 4,                             # quarterly
        'M': 12, 'MS': 12, 'BM': 12, 'BMS': 12, 'CBM': 12, 'CBMS': 12,  # monthly
        'W': 52,                                                        # weekly
        'B': 252, 'C': 252, 'D': 252,                                   # daily
    }

    inferred_freq = getattr(dataframe.index, 'inferred_freq', None)
    freq_num = freq_alias_to_num.get(inferred_freq)
    if freq_num is None:
        raise ValueError('periodicity of given dataframe cannot be defined, '
                         'try to resample data')
    return freq_num
