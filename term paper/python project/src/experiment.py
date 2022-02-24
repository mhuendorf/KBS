# Author: Gideon Vogt
import gosdt
from osdt import bbound
from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
import os.path
# import sys
# sys.path.append('../GeneralizedOptimalSparseDecisionTrees-master/python')
from model.gosdt import GOSDT
import pandas as pd

def run_gosdt(data_csv_paths: [str], config_file_path: str, result_file_path: str, reset_file: bool = False):
    """
    Executes the GOSDT Algorithm on the given Datasets
    Args:
        dataCsvPaths:
        resultFilePath:
        resetFile:

    Returns:

    """
    # reset file or concatenate at the end of it
    file_exists = None
    if os.path.isfile(result_file_path):
        file_exists = True
    else:
        file_exists = False

    if ((not file_exists) or reset_file):
        result_file = open(result_file_path, "w")
    else:
        result_file = open(result_file_path, "a")

    # read and apply config
    with open(config_file_path, "r") as config_file:
        hyperparameters = config_file.read()
    model = GOSDT(hyperparameters)

    # write to result file
    #result_file.write("\nNew Run: \n")
    #result_file.write("Config: \n")
    #result_file.write(hyperparameters)

    # solve datasets given
    for data_csv_path in data_csv_paths:
        # read in data
        dataframe = pd.DataFrame(pd.read_csv(data_csv_path))
        X = dataframe[dataframe.columns[:-1]]
        y = dataframe[dataframe.columns[-1:]]

        # execute
        model.fit(X, y)

        # results
        # TODO Write Function for writing a line of results into resultfile
        exec_time = format(model.time)
        print("Execution Time: {}".format(model.time))
        prediction = model.predict(X)
        training_accuracy = model.score(X, y)
        print("Training Accuracy: {}".format(training_accuracy))
        print(model.tree)
        # write to file
        write_resultline(result_file, "gosdt", data_csv_path, exec_time, str(training_accuracy),
                         str(0))
    result_file.close()

#def run_osdt()

def write_resultline(opened_result_file,
                     algorithm_name: str, dataset_path: str, exec_time: str, train_acc: str, test_acc: str):
    opened_result_file.write(algorithm_name + "," + dataset_path + "," + exec_time + "," + train_acc + "," + test_acc)

if __name__ == '__main__':
    run_gosdt(["../res/test/monk1-train_comma.csv"], "../res/config.json", "../results/first_result_file.csv", False)
