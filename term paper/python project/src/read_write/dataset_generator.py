import random

import pandas as pd

import reader
import os
from random import randrange
from pathlib import Path
import shutil

# the different sizes that should be generated
set_sizes = [50, 100, 200, 500, 1000]

ignore_files = ['dota2Test', 'dota2Train', 'agaricus-lepiota']


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


if __name__ == '__main__':

    # read in the original data and set a seed for consistency
    print('read in files...')
    datasets = reader.collect_data(Path('../../res'))
    random.seed(69420)

    # create folder if not exists
    benchmark_loc = '../../res/benchmarks'
    if not os.path.exists(benchmark_loc):
        os.makedirs(benchmark_loc)

    print('generate benchmark files...')
    completion_counter = 0
    files_to_generate = (len(datasets) - len(ignore_files)) * len(set_sizes)

    # for every dataset create a folder (ignored ota2Test, dota2Train and agaricus-lepiota)
    for name, dataset in datasets:
        dataset_folder = benchmark_loc + '/' + name
        dirpath = Path(dataset_folder)
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
        if name in ignore_files:
            continue
        os.makedirs(dataset_folder)

        # for every size create a file with <size> amount of different randomized cases
        for size in set_sizes:
            c_dataset1, c_dataset2 = sort_and_split_dataset(dataset.copy())
            full_path = dataset_folder + '/' + str(size) + '.csv'
            if os.path.exists(full_path):
                os.remove(full_path)
            with open(full_path, 'w') as outfile:
                print(f'\rgenerated files: {completion_counter}/{files_to_generate}: now generating {outfile.name}',
                      end='')
                if size/2 > len(c_dataset1.index) or size > len(c_dataset2.index):
                    raise AssertionError(f'There is too less data.\n'
                                         f'The first class value {c_dataset1.iloc[0][-1]} has {len(c_dataset1.index)} '
                                         f'entrys ({size/2} entrys needed) and '
                                         f'the second class value {c_dataset1.iloc[0][-1]} has {len(c_dataset2.index)} '
                                         f'entrys ({size/2} entrys needed)')
                outfile.write(','.join(list(c_dataset1.columns.values)) + '\n')
                i = 0
                class_val_1 = True
                while i < size:
                    if class_val_1:
                        c_dataset = c_dataset1
                    else:
                        c_dataset = c_dataset2
                    randint = randrange(int((len(c_dataset.index)-1)/2))
                    unusable = False
                    for entry in c_dataset.iloc[randint]:
                        if str(entry).replace(" ", "") == '?':
                            c_dataset.drop(index=randint, inplace=True)
                            c_dataset.reset_index(inplace=True, drop=True)
                            unusable = True
                            break
                    if unusable:
                        continue
                    if i == size-1:
                        outfile.write(
                            c_dataset.iloc[randint].to_csv(index=False, header=False).replace('\n', ',').replace(' ', '')[:-1])
                    else:
                        outfile.write(
                            c_dataset.iloc[randint].to_csv(index=False, header=False).replace('\n', ',').replace(' ',
                                                                                                               '')[:-1] + '\n')
                    c_dataset.drop(index=randint, inplace=True)
                    c_dataset.reset_index(inplace=True, drop=True)
                    if class_val_1:
                        c_dataset1 = c_dataset
                    else:
                        c_dataset2 = c_dataset
                    i += 1
                    class_val_1 = not class_val_1
            completion_counter += 1
    print('\nfinished')