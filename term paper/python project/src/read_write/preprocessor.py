from pathlib import Path
import pandas as pd
import math
import warnings
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_bool_dtype


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
    else:
        if len(bins) == 0:
            message = 'Column {} has the same value across all cells you should consider deleting it'.format(column)
            warnings.warn(message)
        new_col = []
        for entry in orig_df[column]:
            new_col.append(1 if entry == bins[0] else 0)
        new_df['is-' + column + '?'] = new_col
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


def create_binary_csv(sizes=None, num_bins=5):
    print('preprocess sets...')
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000]
    loc = Path('../../res/benchmarks')
    loc.exists()
    train = Path('train')
    test = Path('test')
    # dirs = [Path('kr-vs-kp')]
    dirs = [Path('adult'), Path('agaricus-lepiota-Reordered'), Path('dota2TrainReordered'),
            Path('kr-vs-kp'), Path('spambase')]
    prefix = 'bin'
    end = '.csv'
    completion_counter = 0
    files_to_generate = len(dirs) * len(sizes) * 2

    for data_set in dirs:
        for number in sizes:
            print(
                f'\rpreprocessed files: {completion_counter}/{files_to_generate}: now generating '
                f'{loc / train / data_set / (prefix + "_" + str(number) + end)}',
                end='')
            bin_and_mask(loc / train / data_set / (str(number) + end),
                         loc / train / data_set / (prefix + '_' + str(number) + end), num_bins)
            completion_counter += 1
            print(
                f'\rpreprocessed files: {completion_counter}/{files_to_generate}: now generating '
                f'{loc / test / data_set / (prefix + "_" + str(number) + end)}',
                end='')
            bin_and_mask(loc / test / data_set / (str(number) + end),
                         loc / test / data_set / (prefix + '_' + str(number) + end),
                         num_bins)
            completion_counter += 1
    print(f'\rpreprocessed files: {completion_counter}/{files_to_generate}')
    print('finished')


if __name__ == '__main__':
    create_binary_csv()
