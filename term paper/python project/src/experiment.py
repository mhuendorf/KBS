# Author: Gideon Vogt
#import gosdt
import osdt
from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
from pyids.model_selection.coordinate_ascent import CoordinateAscent
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
import numpy as np

def run_gosdt_withc(data_csv_paths: [str], config_file_path: str, result_file_path: str, reset_file: bool = False, regularization: float = 0.08):
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
        hyperparameters = config_file.read().replace("\"regularization\": 0.08", "\"regularization\": " + str(regularization))
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
        prediction = model.predict(X)
        training_accuracy = model.score(X, y)
        # write to file
        write_resultline(result_file, "gosdt_c_r" + str(regularization), data_csv_path, exec_time, str(training_accuracy),
                         str(0))
    result_file.close()

def run_gosdt_withoutc(data_csv_paths: [str], config_file_path: str, result_file_path: str, reset_file: bool = False, regularization: float = 0.08):
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
        hyperparameters = config_file.read().replace("\"regularization\": 0.08", "\"regularization\": " + str(regularization))
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
        prediction = model.predict(X)
        training_accuracy = model.score(X, y)
        # write to file
        write_resultline(result_file, "gosdt_noc_r" + str(regularization), data_csv_path, exec_time, str(training_accuracy),
                         str(0))
    result_file.close()

def run_osdt(data_csv_paths: [str], config_file_path: str, result_file_path: str, reset_file: bool = False, regularization: float = 0.001):
    # open file
    result_file = open_resultfile(result_file_path, reset_file)
    for data_csv_path in data_csv_paths:
        data_train = pd.read_csv(data_csv_path)
        X_train = data_train.values[:, :-1]
        y_train = data_train.values[:, -1]

        #regularization
        lamb = regularization
        timelimit = False

        start = time.time()
        # OSDT
        leaves_c, prediction_c, dic, nleaves_OSDT, nrule, ndata, totaltime, time_c, COUNT, C_c, trainaccu_OSDT, best_is_cart, clf = \
            osdt.bbound(X_train, y_train, lamb=lamb, prior_metric="curiosity", timelimit=timelimit, init_cart=True)
        end = time.time()
        exec_time = str(end - start)

        #results
        _, training_accuracy2 = osdt.predict(leaves_c, prediction_c, dic, X_train, y_train, best_is_cart, clf)
        write_resultline(result_file, "osdt_r" + str(regularization), data_csv_path, exec_time, str(training_accuracy2),
                         str(0))
    result_file.close()

def run_pyids(data_csv_paths: [str], config_file_path: str, result_file_path: str, reset_file: bool = False, alg_type: str = "SLS"):
    # open file
    result_file = open_resultfile(result_file_path, reset_file)
    for data_csv_path in data_csv_paths:
        start = time.time()
        data = pd.read_csv(data_csv_path)
        cars = mine_CARs(data, rule_cutoff=50, sample=False)
        #gl = 20.489711934156375
        gl = 1
        lambda_array = [gl,gl,gl,gl,gl,gl,gl]
        quant_dataframe = QuantitativeDataFrame(data)
        quant_cars = list(map(QuantitativeCAR, cars))

        """
        def fmax(lambda_dict):
            print(lambda_dict)
            ids = IDS(algorithm="SLS")
            ids.fit(class_association_rules=cars, quant_dataframe=quant_dataframe, lambda_array=list(lambda_dict.values()))
            auc = ids.score_auc(quant_dataframe)
            print(auc)
            return auc

        coord_asc = CoordinateAscent(
            func=fmax,
            func_args_ranges=dict(
                l1=(1, 1000),
                l2=(1, 1000),
                l3=(1, 1000),
                l4=(1, 1000),
                l5=(1, 1000),
                l6=(1, 1000),
                l7=(1, 1000)
            ),
            ternary_search_precision=50,
            max_iterations=3
        )

        best_lambdas = coord_asc.fit()
        """
        ids = IDS(algorithm=alg_type)  # or SLS DLS DUSM RUSM
        ids.fit(quant_dataframe=quant_dataframe, class_association_rules=cars, lambda_array=lambda_array)

        training_accuracy = ids.score(quant_dataframe)
        print(ids.score_interpretability_metrics(quant_dataframe=quant_dataframe))
        end = time.time()
        exec_time = str(end - start)
        print(ids.clf.rules)
        write_resultline(result_file, "ids_" + alg_type, data_csv_path, exec_time, str(training_accuracy),
                         str(0))
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

def test_regularization(data_csv_paths: [str], result_file_path: str, reg_begin: float, reg_end: float, reg_stepsize: float,
                               run_alg_func:callable, reset_file: bool = False):
    result_file = open_resultfile(result_file_path, reset_file)
    for reg in np.arange(reg_begin, reg_end, reg_stepsize)[::-1]:
        run_alg_func(data_csv_paths,"../res/config.json","../results/reg_test.csv",reset_file,reg)
    result_file.close()

if __name__ == '__main__':
    #test_data = ["../res/test/monk1-train_comma.csv"]
    #test_data = ["../res/test/balance-scale_comma.csv", "../res/test/compas-binary.csv", "../res/adult/bin_500.csv"]
    test_data = ["../res/benchmarks/adult/bin_1000.csv"]
    #test_data = ["../res/benchmarks/spambase/100.csv"]
    #test_data = ["../res/mushroom/agaricus-lepiota.data"]
    #run_gosdt_withc(test_data, "../res/config.json", "../results/first_result_file.csv", False, 0.1)
    #run_gosdt_withoutc(test_data, "../res/config.json", "../results/first_result_file.csv", False, 0.1)
    #run_gosdt_withc(test_data, "../res/config.json", "../results/first_result_file.csv", False, 0.08)
    #run_gosdt_withoutc(test_data, "../res/config.json", "../results/first_result_file.csv", False, 0.08)
    #run_osdt(test_data, "../res/config.json", "../results/first_result_file.csv", False, 0.1)
    #run_osdt(test_data, "../res/config.json", "../results/first_result_file.csv", False, 0.00001)
    #run_pyids(test_data, "../res/config.json", "../results/first_result_file.csv", False)
    test_regularization(test_data, "../res/config.json", 0.08, 0.1, 0.005, run_gosdt_withc,False)
    test_regularization(test_data, "../res/config.json", 0.08, 0.1, 0.005, run_gosdt_withoutc, False)
    test_regularization(test_data, "../res/config.json", 0.08, 0.1, 0.005, run_osdt, False)
