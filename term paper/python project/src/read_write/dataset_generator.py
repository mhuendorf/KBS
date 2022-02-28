import random

from pandas.errors import PerformanceWarning

from new_preprocessor import create_binary_csv
import pandas as pd

import reader
import os
from random import randrange
from pathlib import Path
import shutil
import warnings


#ignore_files = ['dota2Test', 'dota2Train', 'dota2TestReordered', 'agaricus-lepiota']


def sort_and_split_dataset(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    df.sort_values(by=df.columns[len(df.columns) - 1], inplace=True)
    first_val = check_val = df.iloc[0][-1]

    index = 0
    while first_val == check_val:
        index += 1
        check_val = df.iloc[index][-1]

    df1 = df.iloc[:index, :]
    df2 = df.iloc[index:, :]
    df1.reset_index(inplace=True, drop=True)
    df2.reset_index(inplace=True, drop=True)
    return df1, df2


def generate_train_and_test_files(sizes=None):
    """
    Generates all the train and test files such that there are a specific number of cases.
    """
    warnings.simplefilter(action='ignore', category=PerformanceWarning)
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000]
    # read in the original data and set a seed for consistency
    print('read in files...')
    datasets = reader.collect_data(Path('../../res'))
    datasets.sort(key=lambda x: x[0])
    bin_datasets = reader.collect_bin_data(Path('../../res'))
    bin_datasets.sort(key=lambda x: x[0].split('_')[1])
    random.seed(69420)

    # for i in range(len(datasets)):
    #     name, _ = datasets[i]
    #     print(name, end=', ')
    #     name, _ = bin_datasets[i]
    #     print(name)
    # exit()

    # create folder if not exists
    benchmark_train = Path('../../res/benchmarks/train')
    benchmark_test = Path('../../res/benchmarks/test')
    if not os.path.exists(benchmark_train):
        os.makedirs(benchmark_train)
    if not os.path.exists(benchmark_test):
        os.makedirs(benchmark_test)

    print('generate benchmark files...')
    completion_counter = 0
    files_to_generate = (len(datasets)) * len(sizes)

    # for every dataset create a folder (ignored ota2Test, dota2Train and agaricus-lepiota)
    for i in range(len(datasets)):
        name, dataset = datasets[i]
        bin_name, bin_dataset = bin_datasets[i]
        train_dataset_folder = benchmark_train / name
        test_dataset_folder = benchmark_test / name
        if train_dataset_folder.exists() and train_dataset_folder.is_dir():
            shutil.rmtree(train_dataset_folder)
        if test_dataset_folder.exists() and test_dataset_folder.is_dir():
            shutil.rmtree(test_dataset_folder)
        os.makedirs(train_dataset_folder)
        os.makedirs(test_dataset_folder)

        # for every size create a file with <size> amount of different randomized cases
        for size in sizes:
            d = dataset.copy()
            for col in bin_dataset.columns:
                d.insert(len(d.columns), col, bin_dataset[col])

            c1, c2 = sort_and_split_dataset(d)
            c_bin_dataset1 = pd.DataFrame()
            c_bin_dataset2 = pd.DataFrame()
            for col in range(len(bin_dataset.columns)):
                c_bin_dataset1.insert(0, c1.columns[len(c1.columns)-1], c1.pop(c1.columns[len(c1.columns)-1]))
                c_bin_dataset2.insert(0, c2.columns[len(c2.columns)-1], c2.pop(c2.columns[len(c2.columns)-1]))
            c_dataset1 = c1.copy()
            c_dataset2 = c2.copy()

            train_full_path = (train_dataset_folder / str(size)).with_suffix('.csv')
            bin_train_full_path = (train_dataset_folder / ('bin_' + str(size))).with_suffix('.csv')
            test_full_path = (test_dataset_folder / str(size)).with_suffix('.csv')
            bin_test_full_path = (test_dataset_folder / ('bin_' + str(size))).with_suffix('.csv')
            if os.path.exists(train_full_path):
                os.remove(train_full_path)
            with open(train_full_path, 'w') as outfile:
                with open(bin_train_full_path, 'w') as bin_outfile:
                    print(f'\rgenerated files: {completion_counter}/{files_to_generate}: now generating {outfile.name}',
                          end='')
                    if size / 2 > len(c_dataset1.index) or size > len(c_dataset2.index):
                        raise AssertionError(f'There is too less data.\n'
                                             f'The first class value {c_dataset1.iloc[0][-1]} has {len(c_dataset1.index)} '
                                             f'entrys ({size / 2} entrys needed) and '
                                             f'the second class value {c_dataset1.iloc[0][-1]} has {len(c_dataset2.index)} '
                                             f'entrys ({size / 2} entrys needed)')
                    outfile.write((','.join(list(c_dataset1.columns.values))).replace(" ", "") + '\n')
                    bin_outfile.write((','.join(list(c_bin_dataset1.columns.values))).replace(" ", "") + '\n')

                    i = 0
                    class_val_1 = True
                    while i < size:

                        # alternate class values
                        if class_val_1:
                            c_bin_dataset = c_bin_dataset1
                            c_dataset = c_dataset1
                        else:
                            c_bin_dataset = c_bin_dataset2
                            c_dataset = c_dataset2

                        # choose a random case and write it into the file
                        randint = randrange(int((len(c_dataset.index) - 1) / 2))
                        unusable = False
                        for entry in c_dataset.iloc[randint]:
                            if str(entry).replace(" ", "") == '?':
                                c_dataset.drop(index=randint, inplace=True)
                                c_bin_dataset.drop(index=randint, inplace=True)
                                c_dataset.reset_index(inplace=True, drop=True)
                                c_bin_dataset.reset_index(inplace=True, drop=True)
                                unusable = True
                                break
                        if unusable:
                            continue
                        if i == size - 1:
                            outfile.write(
                                c_dataset.iloc[randint].to_csv(index=False, header=False).replace('\n', ',').replace(' ',
                                                                                                                     '')[
                                :-1])
                            bin_outfile.write(
                                c_bin_dataset.iloc[randint].to_csv(index=False, header=False).replace('\n', ',').replace(
                                    ' ',
                                    '')[
                                :-1])
                        else:
                            outfile.write(
                                c_dataset.iloc[randint].to_csv(index=False, header=False).replace('\n', ',').replace(' ',
                                                                                                                     '')[
                                :-1] + '\n')
                            bin_outfile.write(
                                c_bin_dataset.iloc[randint].to_csv(index=False, header=False).replace('\n',
                                                                                                      ',').replace(
                                    ' ',
                                    '')[
                                :-1] + '\n')

                        c_dataset.drop(index=randint, inplace=True)
                        c_bin_dataset.drop(index=randint, inplace=True)
                        c_dataset.reset_index(inplace=True, drop=True)
                        c_bin_dataset.reset_index(inplace=True, drop=True)

                        # change class value selection
                        if class_val_1:
                            c_dataset1 = c_dataset
                            c_bin_dataset1 = c_bin_dataset
                        else:
                            c_dataset2 = c_dataset
                            c_bin_dataset2 = c_bin_dataset
                        i += 1
                        class_val_1 = not class_val_1

            # create test file
            test_set = pd.concat([c_dataset1, c_dataset2], ignore_index=True)
            bin_test_set = pd.concat([c_bin_dataset1, c_bin_dataset2], ignore_index=True)

            """for index, row in test_set.iterrows():
                ifor_val = something
                if < condition >:
                    ifor_val = something_else
                df.set_value(i, 'ifor', ifor_val)"""


            # remove all undefined cases
            for index, row in test_set.iterrows():
                j = 0
                for value in row:
                    c_val = str(value).replace(' ', '')
                    if c_val == "?":
                        test_set.drop(index=index, inplace=True)
                        bin_test_set.drop(index=index, inplace=True)
                        break
                    else:
                        test_set.at[index, test_set.columns[j]] = c_val
                    j += 1

            new_columns = {}
            for column in test_set.columns:
                new_columns[column] = str(column).replace(' ', '')
            test_set.rename(columns=new_columns, inplace=True)

            new_columns = {}
            for column in bin_test_set.columns:
                new_columns[column] = str(column).replace(' ', '')
            bin_test_set.rename(columns=new_columns, inplace=True)

            test_set.reset_index(inplace=True, drop=True)
            bin_test_set.reset_index(inplace=True, drop=True)
            test_set.to_csv(test_full_path, index=False)
            bin_test_set.to_csv(bin_test_full_path, index=False)

            completion_counter += 1
    print(f'\rgenerated files: {completion_counter}/{files_to_generate}: now generating {outfile.name}', end='')
    print('\nfinished')


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=PerformanceWarning)
    set_sizes = [50, 100, 200, 500, 1000]
    create_binary_csv(5)
    generate_train_and_test_files(set_sizes)
