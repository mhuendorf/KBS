# This file is used to read in the datasets
from pathlib import Path

import pandas as pd
from os import listdir
from os.path import isfile, join, splitext

# static variable to indicate all the datasets folders
categories = ['census', 'chess', 'dota', 'mushroom', 'spam']


def collect_data(path_to_resorces: Path):
    """Reads in every datafile in the specified folders and returns it."""
    datasets = []
    for folder in categories:
        path = path_to_resorces / folder
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for file in files:
            if splitext(file)[1] == '.data' or splitext(file)[1] == '.csv':
                datasets.append((splitext(file)[0], pd.read_csv(path / file)))

    return datasets
