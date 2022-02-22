import random

import reader
import os
from random import randrange
from pathlib import Path
import shutil

# the different sizes that should be generated
set_sizes = [50, 100, 200, 500, 1000]

ignore_files = ['dota2Test', 'dota2Train', 'agaricus-lepiota']

if __name__ == '__main__':

    # read in the original data and set a seed for consistency
    print('read in files...')
    datasets = reader.collect_data('../../res')
    random.seed(402)

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
            c_dataset = dataset.copy()
            full_path = dataset_folder + '/' + str(size) + '.csv'
            if os.path.exists(full_path):
                os.remove(full_path)
            with open(full_path, 'w') as outfile:
                if size > len(c_dataset.index):
                    raise AssertionError
                outfile.write(','.join(list(c_dataset.columns.values)) + '\n')
                for i in range(size):
                    randint = randrange(len(c_dataset.index)-1)
                    if i == size:
                        outfile.write(
                            c_dataset.iloc[randint].to_csv(index=False, header=False).replace('\n', ',').replace(' ', ''))
                    else:
                        outfile.write(
                            c_dataset.iloc[randint].to_csv(index=False, header=False).replace('\n', ',').replace(' ',
                                                                                                               '') + '\n')
                    c_dataset.drop(index=randint, inplace=True)
                    c_dataset.reset_index(inplace=True, drop=True)
            completion_counter += 1
            print(f'\rgenerated files: {completion_counter}/{files_to_generate}', end='')
    print('\nfinished')


