# Author: Gideon Vogt
#import gosdt
import osdt
from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
import os.path
# import sys
# sys.path.append('../GeneralizedOptimalSparseDecisionTrees-master/python')
import gosdt
from model.gosdt import GOSDT
import pandas as pd
import time
from pyarc.qcba.data_structures import (
QuantitativeDataFrame,
QuantitativeCAR
)

def run_gosdt_withc(data_csv_paths: [str], config_file_path: str, result_file_path: str, reset_file: bool = False):
    """
    Executes the GOSDT Algorithm on the given Datasets
    Args:
        dataCsvPaths:
        resultFilePath:
        resetFile:

    Returns:

    """
    # open file
    result_file = open_resultfile(result_file_path, reset_file)

    # read and apply config
    with open(config_file_path, "r") as config_file:
        hyperparameters = config_file.read()
    model = GOSDT(hyperparameters)
    gosdt.configure(hyperparameters)

    # write to result file
    #result_file.write("\nNew Run: \n")
    #result_file.write("Config: \n")
    #result_file.write(hyperparameters)

    # solve datasets given
    for data_csv_path in data_csv_paths:
        # make model with c++
        with open(data_csv_path, "r") as data_file:
            data = data_file.read()
        start = time.time()
        result = gosdt.fit(data)
        model_file = open("../results/model.json", "w")
        model_file.write(result)
        model_file.close()

        # read in data
        dataframe = pd.DataFrame(pd.read_csv(data_csv_path))
        X = dataframe[dataframe.columns[:-1]]
        y = dataframe[dataframe.columns[-1:]]

        # execute

        #model.fit(X, y)
        model.load("../results/model.json")
        end = time.time()
        exec_time = str(end - start)

        # results
        # TODO Write Function for writing a line of results into resultfile
        prediction = model.predict(X)
        training_accuracy = model.score(X, y)
        # write to file
        write_resultline(result_file, "gosdt_c", data_csv_path, exec_time, str(training_accuracy),
                         str(0))
    result_file.close()

def run_gosdt_withoutc(data_csv_paths: [str], config_file_path: str, result_file_path: str, reset_file: bool = False):
    """
    Executes the GOSDT Algorithm on the given Datasets
    Args:
        dataCsvPaths:
        resultFilePath:
        resetFile:

    Returns:

    """
    # open file
    result_file = open_resultfile(result_file_path, reset_file)

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

        start = time.time()
        model.fit(X, y)
        end = time.time()
        exec_time = str(end - start)

        # results
        # TODO Write Function for writing a line of results into resultfile
        prediction = model.predict(X)
        training_accuracy = model.score(X, y)
        # write to file
        write_resultline(result_file, "gosdt_noc", data_csv_path, exec_time, str(training_accuracy),
                         str(0))
    result_file.close()

def run_osdt(data_csv_paths: [str], config_file_path: str, result_file_path: str, reset_file: bool = False):
    # open file
    result_file = open_resultfile(result_file_path, reset_file)
    for data_csv_path in data_csv_paths:
        data_train = pd.read_csv(data_csv_path)
        X_train = data_train.values[:, :-1]
        y_train = data_train.values[:, -1]

        #regularization
        lamb = 0.05
        timelimit = True

        start = time.time()
        # OSDT
        leaves_c, prediction_c, dic, nleaves_OSDT, nrule, ndata, totaltime, time_c, COUNT, C_c, trainaccu_OSDT, best_is_cart, clf = \
            osdt.bbound(X_train, y_train, lamb=lamb, prior_metric="curiosity", timelimit=timelimit, init_cart=True)
        end = time.time()
        exec_time = str(end - start)

        #results
        _, training_accuracy2 = osdt.predict(leaves_c, prediction_c, dic, X_train, y_train, best_is_cart, clf)
        write_resultline(result_file, "osdt", data_csv_path, exec_time, str(training_accuracy2),
                         str(0))
    result_file.close()

def run_pyids(data_csv_paths: [str], config_file_path: str, result_file_path: str, reset_file: bool = False):
    # open file
    result_file = open_resultfile(result_file_path, reset_file)
    for data_csv_path in data_csv_paths:
        data = pd.read_csv(data_csv_path)
        cars = mine_CARs(data,50)
        lambda_array = [1,1,1,1,1,1,1]
        quant_dataframe = QuantitativeDataFrame(data)
        quant_cars = list(map(QuantitativeCAR, cars))
        ids = IDS()
        ids.fit(quant_dataframe=quant_dataframe, class_association_rules=cars, lambda_array=lambda_array)
        training_accuracy = ids.score(quant_dataframe)
        #TODO WRITE TO FILE
    result_file.close()

def open_resultfile(result_file_path: str, reset_file: bool = False):
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
    return result_file


def write_resultline(opened_result_file,
                     algorithm_name: str, dataset_path: str, exec_time: str, train_acc: str, test_acc: str):
    opened_result_file.write(algorithm_name + "," + dataset_path + "," + exec_time + "," + train_acc + "," + test_acc + "\n")

if __name__ == '__main__':
    #test_data = ["../res/test/monk1-train_comma.csv"]
    #test_data = ["../res/test/balance-scale_comma.csv", "../res/test/compas-binary.csv", "../res/adult/bin_500.csv"]
    test_data = ["../res/benchmarks/adult/bin_200.csv"]
    run_gosdt_withc(test_data, "../res/config.json", "../results/first_result_file.csv", False)
    run_gosdt_withoutc(test_data, "../res/config.json", "../results/first_result_file.csv", False)
    run_osdt(test_data, "../res/config.json", "../results/first_result_file.csv", False)
