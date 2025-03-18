import pandas as pd
from math import factorial
from itertools import combinations
import logging


def find_msu(dataframe, groups, att):
    """
    Find and score each Minimal Sample Unique (MSU) within the dataframe
    for the specified groups
    :param dataframe: the complete dataframe of data to score
    :param groups: an array of arrays for each group of columns to test for uniqueness
    :param att: the total number of attributes (QIDs) in the dataset
    :return:
    """
    df_copy = dataframe
    # 'nple' as we may be testing a group that's a single, a tuple, triple etc
    df_updates = []
    for nple in groups:
        nple = list(nple)
        cols = nple.copy()

        # Calculate the unique value counts (fK)
        cols.append('fK')
        value_counts = df_copy[nple].groupby(nple, sort=False).size()

        if 1 in value_counts.values:
            df_value_counts = pd.DataFrame(value_counts)
            df_value_counts = df_value_counts.reset_index()
            # Change the column names
            df_value_counts.columns = cols

            # Add values for fM, MSU and SUDA
            df_value_counts['msu'] = 0
            df_value_counts.loc[df_value_counts['fK'] == 1, ['msu']] = \
                [str(cols[:-1])]

            # Collect results
            df_update = pd.merge(df_copy, df_value_counts, on=nple, how='left')
            df_updates.append(df_update)

    # Return results
    if len(df_updates) > 0:
        df_updates = pd.concat(df_updates)
    return df_updates

def get_minimal_msu(x):
    for msu in x:
        if msu:
            return msu
    return 0

def suda2(dataframe, max_msu=3, dis=0.1, columns=None):
    """
    Special Uniqueness Detection Algorithm (SUDA)
    :param dataframe:
    :param max_msu:
    :param dis:
    :param columns: the set of columns to apply SUDA to. Defaults to None (all columns)
    :return:
    """
    logger = logging.getLogger("suda")
    logging.basicConfig()

    # Get the set of columns
    if columns is None:
        columns = dataframe.columns

    cols = dataframe.columns[dataframe.nunique() < 600]
    dataframe[cols] = dataframe[cols].apply(lambda x: x.astype(pd.CategoricalDtype(ordered=True)))

    att = len(columns)
    if att > 20:
        logger.warning("More than 20 columns presented; setting ATT to max of 20")
        att = 20

    # Construct the aggregation array
    aggregations = {'msu': lambda x: get_minimal_msu(x), 'fK': 'min'}
    for column in dataframe.columns:
        aggregations[column] = 'max'

    results = []
    for i in range(1, max_msu+1):
        groups = list(combinations(columns, i))
        result = (find_msu(dataframe, groups, att))
        if len(result) != 0:
            results.append(result)

    if len(results) == 0:
        logger.info("No special uniques found")
        dataframe["msu"] = None
        dataframe['fK'] = None
        return dataframe

    # Concatenate all results
    results.append(dataframe)
    results = pd.concat(results).groupby(level=0).agg(aggregations)

    results['msu'] = results['msu'].fillna(0)
    return results['msu']


