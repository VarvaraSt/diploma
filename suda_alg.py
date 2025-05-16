import pandas as pd
from itertools import combinations
import logging


def find_msu(dataframe, groups, k=1):
    df_copy = dataframe
    df_updates = []
    for nple in groups:
        nple = list(nple)
        cols = nple.copy()

        cols.append('fK')
        value_counts = df_copy[nple].groupby(nple, sort=False).size()

        if any(value_counts <= k):
            df_value_counts = pd.DataFrame(value_counts)
            df_value_counts = df_value_counts.reset_index()
            df_value_counts.columns = cols

            df_value_counts['msu'] = 0
            df_value_counts.loc[df_value_counts['fK'] <= k, ['msu']] = \
                [str(cols[:-1])]

            df_update = pd.merge(df_copy, df_value_counts, on=nple, how='left')
            df_updates.append(df_update)

    if len(df_updates) > 0:
        df_updates = pd.concat(df_updates)
    return df_updates

def get_minimal_msu(x):
    for msu in x:
        if msu:
            return msu
    return 0

def suda2(dataframe, max_msu=3, columns=None, k=1):
    logger = logging.getLogger("suda")
    logging.basicConfig()

    if columns is None:
        columns = dataframe.columns

    cols = dataframe.columns[dataframe.nunique() < 600]
    dataframe[cols] = dataframe[cols].apply(lambda x: x.astype(pd.CategoricalDtype(ordered=True)))

    aggregations = {'msu': lambda x: get_minimal_msu(x), 'fK': 'min'}
    for column in dataframe.columns:
        aggregations[column] = 'max'

    results = []
    for i in range(1, max_msu+1):
        groups = list(combinations(columns, i))
        result = (find_msu(dataframe, groups, k))
        if len(result) != 0:
            results.append(result)

    if len(results) == 0:
        logger.info("No special uniques found")
        dataframe["msu"] = None
        dataframe['fK'] = None
        return None

    results.append(dataframe)
    results = pd.concat(results).groupby(level=0).agg(aggregations)

    results['msu'] = results['msu'].fillna(0)
    return results['msu']


