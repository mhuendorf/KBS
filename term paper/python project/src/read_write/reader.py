# This file is used to read in the datasets
import pandas as pd
from os import listdir
from os.path import isfile, join, splitext

# static variable to indicate all the datasets folders
categories = ['census', 'chess', 'dota', 'mushroom', 'spam']


def collect_data():
    """Reads in every datafile in the specified folders and returns it."""
    datasets = []
    for folder in categories:
        path = '../res/' + folder
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for file in files:
            if splitext(file)[1] == '.data' or splitext(file)[1] == '.csv':
                datasets.append(pd.read_csv(path + '/' + file))

    return datasets

