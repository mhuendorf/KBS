from pathlib import Path
import pandas as pd
import math
import warnings
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_bool_dtype


def handle_numeric(orig_df: pd.DataFrame, new_df: pd.DataFrame, column: str, num_bins: int) -> pd.DataFrame:
    """
    Handles all numeric columns.
    Args:
        orig_df: The original DataFrame
        new_df: The new DataFrame where the new column is being inserted into
        column: The column to be binned and made binary attributes
        num_bins: The number of bin that should be used

    Returns: The new DataFrame where the new column is being inserted into
    """

    # get the minimal and maximum value of the column
    min_val = math.inf
    max_val = -math.inf
    values = []
    for entry in orig_df[column]:
        min_val = min(min_val, entry)
        max_val = max(max_val, entry)
        if entry not in values:
            values.append(entry)

    # if there are too less different values they may be viewed as categories
    if len(values) < num_bins:
        return handle_categories(orig_df, new_df, column)
    else:
        # create limits of bins
        bins = []
        for i in range(num_bins):
            bin_val = (min_val * (num_bins + 1) + max_val * (i + 1) - min_val * (i + 1)) / (num_bins + 1)
            if max_val - min_val < 10:
                bins.append(bin_val)
            else:
                bins.append(int(bin_val))

        # create the new columns
        for bin_val in bins:
            new_col = []
            for i in range(len(orig_df[column])):
                if orig_df[column].values[i] >= bin_val:
                    new_col.append(1)
                else:
                    new_col.append(0)
            new_df[column + '>=' + str(bin_val)] = new_col
            new_df = new_df.copy()
    return new_df


def handle_categories(orig_df: pd.DataFrame, new_df: pd.DataFrame, column: str):
    """
    Handle all categorised columns.
    Args:
        orig_df: The original DataFrame
        new_df: The new DataFrame where the new column is being inserted into
        column: The column to be binned and made binary attributes

    Returns: The new DataFrame where the new column is being inserted into
    """

    # create a bin for every different value
    bins = []
    for entry in orig_df[column]:
        if entry not in bins:
            bins.append(entry)

    # for every category create a column and insert its binary truth values
    if len(bins) > 2:
        for bin_val in bins:
            new_col = []
            for i in range(len(orig_df[column])):
                if orig_df[column].values[i] == bin_val:
                    new_col.append(1)
                else:
                    new_col.append(0)
            new_df[column + '-' + str(bin_val)] = new_col

    # if categories are binary handle them as such
    elif len(bins) == 2:
        new_col = []
        for entry in orig_df[column]:
            new_col.append(1 if entry == bins[0] else 0)
        new_df['is-' + str(bins[0]) + '-not-' + str(bins[1])] = new_col
    # if every value of a column is tha same its attribute has no relevance, so it gets discarded
    # show a warning so the user is warned by the maybe unintended behavior
    else:
        warnings.warn(f"Column {column} has the same value across all cells and therefore gets deleted")
    return new_df.copy()


def handle_bools(orig_df: pd.DataFrame, new_df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Handle all binary columns.
    Args:
        orig_df: The original DataFrame
        new_df: The new DataFrame where the new column is being inserted into
        column: The column to be binned and made binary attributes

    Returns: The new DataFrame where the new column is being inserted into
    """
    new_col = []
    for entry in orig_df[column]:
        new_col.append(1 if entry else 0)
    new_df[column] = new_col
    return new_df.copy()


def bin_and_mask(path_from: Path, path_to: Path, num_bins_numeric: int):
    """
    Given a csv-style file, this function bins numeric values of the same attribute and splits all columns into
    multiple new columns, such that every entry in the whole new table is binary.
    Args:
        path_from: The path to the file that is being read
        path_to: The path where the resulting table should be saved to
        num_bins_numeric: The number of bins that should be created for all numeric attributes
    """
    df = pd.read_csv(path_from, encoding='utf-8', sep=",")
    new_df = pd.DataFrame()
    for column in df.columns:
        # if number, bin with range
        if is_numeric_dtype(df[column]):
            new_df = handle_numeric(df, new_df, column, num_bins_numeric)
        elif is_string_dtype(df[column]):
            new_df = handle_categories(df, new_df, column)
        elif is_bool_dtype(df[column]):
            new_df = handle_bools(df, new_df, column)
        else:
            raise TypeError(f"type <{df[column].dtype}> is not supported")
    new_df.to_csv(path_to, index=False)


if __name__ == '__main__':
    numbers = [50, 100, 200, 500, 1000]
    loc = Path('../../res/benchmarks')
    loc.exists()
    dirs = [Path('adult'), Path('agaricus-lepiota-Reordered'), Path('dota2TestReordered'), Path('dota2TrainReordered'), Path('kr-vs-kp'), Path('spambase')]
    prefix = 'bin'
    number = '500'
    end = '.csv'
    for dir in dirs:
        for number in numbers:
            print(loc / dir / (str(number) + end))
            bin_and_mask(loc / dir / (str(number) + end), loc / dir / (prefix + '_' + str(number) + end), 10)
