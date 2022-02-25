# This file is used to read in the datasets
from pathlib import Path

import pandas as pd
from os import listdir
from os.path import isfile, join, splitext

# static variable to indicate all the datasets folders
categories = ['census', 'chess', 'dota', 'mushroom', 'spam']
ignore_files = ['dota2Test.csv', 'dota2Train.csv', 'dota2TestReordered.csv', 'agaricus-lepiota.data']


def collect_data(path_to_resources: Path) -> [(str, pd.DataFrame)]:
    """Reads in every datafile in the specified folders and returns it."""
    datasets = []
    for folder in categories:
        path = path_to_resources / folder
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for file in files:
            if splitext(file)[1] == '.data' or splitext(file)[1] == '.csv':
                if file not in ignore_files:
                    datasets.append((splitext(file)[0], pd.read_csv(path / file)))

    return datasets


def collect_bin_data(path_to_resources: Path) -> [(str, pd.DataFrame)]:
    if not (path_to_resources / 'bin').exists():
        raise FileNotFoundError()
    datasets = []
    files = [f for f in listdir(path_to_resources / 'bin') if isfile(join(path_to_resources / 'bin', f))]
    for file in files:
        datasets.append((splitext(file)[0], pd.read_csv(path_to_resources / 'bin' / file)))

    return datasets


if __name__ == '__main__':
    data = collect_data(Path('../../res'))
    for name, df in data:
        print(f'{name}:')
        print(df)

    data = collect_bin_data(Path('../../res'))
    for name, df in data:
        print(f'{name}:')
        print(df)