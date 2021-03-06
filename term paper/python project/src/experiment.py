# Author: Gideon Vogt
# import gosdt
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


def run_gosdt_withc(data_csv_paths: [str], test_csv_paths: [str], config_file_path: str, result_file_path: str,
                    reset_file: bool = False, regularization: float = 0.08):
    """
    Function that runs gosdt algorithm with its c++ extension
    Args:
        data_csv_paths: paths to train data csv files
        test_csv_paths: paths to corresponding data csv files
        config_file_path: path to config file of gosdt
        result_file_path: path to which results are written
        reset_file: whether result file should be reset
        regularization: value of regularization for this algorithm

    Returns: nothing, only writes to file

    """
    # open file
    result_file = open_resultfile(result_file_path, reset_file)

    # read and apply config
    with open(config_file_path, "r") as config_file:
        hyperparameters = config_file.read().replace("\"regularization\": 0.08",
                                                     "\"regularization\": " + str(regularization))
    model = GOSDT(hyperparameters)
    gosdt.configure(hyperparameters)

    # write to result file
    # result_file.write("\nNew Run: \n")
    # result_file.write("Config: \n")
    # result_file.write(hyperparameters)

    # solve datasets given
    for i in range(len(data_csv_paths)):
        # make model with c++
        with open(data_csv_paths[i], "r") as data_file:
            data = data_file.read()
        start = time.time()
        result = gosdt.fit(data)
        model_file = open("../results/model.json", "w")
        model_file.write(result)
        model_file.close()

        # execute

        # model.fit(X, y)
        model.load("../results/model.json")
        end = time.time()
        exec_time = str(end - start)

        # training acc
        dataframe = pd.DataFrame(pd.read_csv(data_csv_paths[i]))
        X = dataframe[dataframe.columns[:-1]]
        y = dataframe[dataframe.columns[-1:]]
        # results
        # prediction = model.predict(X)
        training_accuracy = model.score(X, y)

        # test acc
        dataframe = pd.DataFrame(pd.read_csv(test_csv_paths[i]))
        X = dataframe[dataframe.columns[:-1]]
        y = dataframe[dataframe.columns[-1:]]
        # results
        # prediction = model.predict(X)
        test_accuracy = model.score(X, y)

        # write to file
        write_resultline(result_file, "gosdt_c_r", data_csv_paths[i], exec_time, str(training_accuracy),
                         str(test_accuracy), str(regularization))
    result_file.close()


def run_gosdt_withoutc(data_csv_paths: [str], test_csv_paths: [str], config_file_path: str, result_file_path: str,
                       reset_file: bool = False, regularization: float = 0.08):
    """
    Function that runs gosdt algorithm without its c++ extension
    Args:
        data_csv_paths: paths to train data csv files
        test_csv_paths: paths to corresponding data csv files
        config_file_path: path to config file of gosdt
        result_file_path: path to which results are written
        reset_file: whether result file should be reset
        regularization: value of regularization for this algorithm

    Returns: nothing, only writes to file

    """
    # open file
    result_file = open_resultfile(result_file_path, reset_file)

    with open(config_file_path, "r") as config_file:
        hyperparameters = config_file.read().replace("\"regularization\": 0.08",
                                                     "\"regularization\": " + str(regularization))
    # model = GOSDT(hyperparameters)
    gosdt.configure(hyperparameters)

    # read and apply config
    # with open(config_file_path, "r") as config_file:
    # hyperparameters = config_file.read().replace("\"regularization\": 0.08", "\"regularization\": " + str(regularization))
    hyperparameters = {
        "regularization": regularization,
        "time_limit": 0,
        "verbose": True,
    }
    model = GOSDT(hyperparameters)

    # write to result file
    # result_file.write("\nNew Run: \n")
    # result_file.write("Config: \n")
    # result_file.write(hyperparameters)

    # solve datasets given
    for i in range(len(data_csv_paths)):
        data_csv_path = data_csv_paths[i]
        # read in data
        dataframe = pd.DataFrame(pd.read_csv(data_csv_path))
        X = dataframe[dataframe.columns[:-1]]
        y = dataframe[dataframe.columns[-1:]]

        print(X)
        print(y)

        # execute

        start = time.time()
        model.fit(X, y)
        end = time.time()
        exec_time = str(end - start)

        # results
        # prediction = model.predict(X)
        training_accuracy = model.score(X, y)

        # test acc
        dataframe = pd.DataFrame(pd.read_csv(test_csv_paths[i]))
        X = dataframe[dataframe.columns[:-1]]
        y = dataframe[dataframe.columns[-1:]]
        # results
        # prediction = model.predict(X)
        test_accuracy = model.score(X, y)

        print(model.tree)
        # write to file
        write_resultline(result_file, "gosdt_noc_r", data_csv_path, exec_time, str(training_accuracy),
                         str(test_accuracy), str(regularization))
    result_file.close()


def run_osdt(data_csv_paths: [str], test_csv_paths: [str], config_file_path: str, result_file_path: str,
             reset_file: bool = False, regularization: float = 0.001):
    """
        Function that runs osdt
        Args:
            data_csv_paths: paths to train data csv files
            test_csv_paths: paths to corresponding data csv files
            config_file_path: path to config file of gosdt
            result_file_path: path to which results are written
            reset_file: whether result file should be reset
            regularization: value of regularization for this algorithm

        Returns: nothing, only writes to file

    """
    # open file
    result_file = open_resultfile(result_file_path, reset_file)
    for i in range(len(data_csv_paths)):
        data_csv_path = data_csv_paths[i]
        data_train = pd.read_csv(data_csv_path)
        X_train = data_train.values[:, :-1]
        y_train = data_train.values[:, -1]
        data_test = pd.read_csv(test_csv_paths[i])
        X_test = data_test.values[:, :-1]
        y_test = data_test.values[:, -1]

        # regularization
        lamb = regularization
        timelimit = False

        start = time.time()
        # OSDT
        leaves_c, prediction_c, dic, nleaves_OSDT, nrule, ndata, totaltime, time_c, COUNT, C_c, trainaccu_OSDT, best_is_cart, clf = \
            osdt.bbound(X_train, y_train, lamb=lamb, prior_metric="curiosity", timelimit=timelimit, init_cart=True)
        end = time.time()
        exec_time = str(end - start)

        # results
        _, training_accuracy2 = osdt.predict(leaves_c, prediction_c, dic, X_train, y_train, best_is_cart, clf)
        _, test_accuracy2 = osdt.predict(leaves_c, prediction_c, dic, X_test, y_test, best_is_cart, clf)
        write_resultline(result_file, "osdt_r", data_csv_path, exec_time, str(training_accuracy2),
                         str(test_accuracy2), str(regularization))
    result_file.close()


def run_pyids(data_csv_paths: [str], test_csv_paths: [str], config_file_path: str, result_file_path: str,
              reset_file: bool = False, alg_type: str = "SLS"):
    """
            Function that runs pyids
            Args:
                data_csv_paths: paths to train data csv files
                test_csv_paths: paths to corresponding data csv files
                config_file_path: path to config file of gosdt (unused here)
                result_file_path: path to which results are written
                reset_file: whether result file should be reset
                alg_type: which type of PyIDS algorithm should be used

            Returns: nothing, only writes to file

    """
    # open file
    result_file = open_resultfile(result_file_path, reset_file)
    for i in range(len(data_csv_paths)):
        data_csv_path = data_csv_paths[i]
        start = time.time()
        data = pd.read_csv(data_csv_path)
        cars = mine_CARs(data, rule_cutoff=50, sample=False)
        # gl = 20.489711934156375
        gl = 1
        lambda_array = [gl, gl, gl, gl, gl, gl, gl]
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

        data_test = pd.read_csv(test_csv_paths[i])
        quant_dataframe_test = QuantitativeDataFrame(data_test)
        test_accuracy = ids.score(quant_dataframe_test)

        print(ids.score_interpretability_metrics(quant_dataframe=quant_dataframe))
        end = time.time()
        exec_time = str(end - start)
        print(ids.clf.rules)
        write_resultline(result_file, "ids_", data_csv_path, exec_time, str(training_accuracy),
                         str(test_accuracy), alg_type)
    result_file.close()


def open_resultfile(result_file_path: str, reset_file: bool = False):
    """
    Opens result file and returns it
    Args:
        result_file_path: Path to file
        reset_file: Wether file content should be reset

    Returns: Opened file

    """
    # reset file or concatenate at the end of it
    file_exists = None
    if os.path.isfile(result_file_path):
        file_exists = True
    else:
        file_exists = False

    if (file_exists and reset_file):
        os.remove(result_file_path)

    if ((not file_exists) or reset_file):
        result_file = open(result_file_path, "w")
    else:
        result_file = open(result_file_path, "a")
    return result_file


def write_resultline(opened_result_file,
                     algorithm_name: str, dataset_path: str, exec_time: str, train_acc: str, test_acc: str,
                     regularization: str = "0"):
    """
    Function that writes some values to a file in csv format
    Args:
        opened_result_file: Opened file to write to
        algorithm_name: Name of the algorithm used
        dataset_path: Path to used dataset
        exec_time: Execution time
        train_acc: Training accuracy
        test_acc: Test accuracy

    Returns: Nothing, only writes to file

    """
    opened_result_file.write(
        algorithm_name + "," + regularization + "," + dataset_path + "," + exec_time + "," + train_acc + "," + test_acc + "\n")


def test_regularization(data_csv_paths: [str], test_csv_paths: [str], config_file_path: str, result_file_path: str,
                        reg_begin: float, reg_end: float, reg_stepsize: float,
                        run_alg_func: callable, reset_file: bool = False, factor: bool = False):
    """
    Tests the given algorithm with different values for regularization and writes it to file
    Args:
        data_csv_paths: paths to train data csv files
        test_csv_paths: paths to corresponding data csv files
        config_file_path: path to config file of gosdt
        reg_begin: Begin of regularization interval
        reg_end: End of interval
        reg_stepsize: Stepsize used inside interval
        run_alg_func: Function to run (gosdt or osdt)
        reset_file: Whether files content should be reset
        factor: Whether Stepsize should be used as a factor, meaning one does not go in steps of 0.01 which are added
        but instead with a stepsize of 10 one would go from reg_begin to 10*reg_begin to 100*reg_begin until
        the end is reached

    Returns: Nothing, only writes to file.

    """
    result_file = open_resultfile(result_file_path, reset_file)
    if (factor):
        reg = reg_end
        while reg >= reg_begin:
            run_alg_func(data_csv_paths, test_csv_paths, config_file_path, result_file_path, reset_file=False,
                         regularization=reg)
            reg = reg / reg_stepsize
    else:
        for reg in np.arange(reg_begin, reg_end, reg_stepsize)[::-1]:
            run_alg_func(data_csv_paths, test_csv_paths, config_file_path, result_file_path, reset_file=False,
                         regularization=reg)
    result_file.close()


def testreg_osdt():
    """
    Runs the tests for regularization for osdt
    Returns: Nothing

    """
    train_data = ["../res/benchmarks/train/kr-vs-kp/bin_1000.csv"]
    test_data = ["../res/benchmarks/test/kr-vs-kp/bin_1000.csv"]
    test_regularization(train_data, test_data, "../res/config.json", "../results/reg_test_osdt.csv", 0.0000000001, 0.2,
                        1.5,
                        run_osdt, reset_file=True, factor=True)


def testreg_gosdt():
    """
    Runs the tests for regularization for gosdt
    Returns: nothing

    """
    train_data = ["../res/benchmarks/train/kr-vs-kp/bin_1000.csv"]
    test_data = ["../res/benchmarks/test/kr-vs-kp/bin_1000.csv"]
    test_regularization(train_data, test_data, "../res/config.json", "../results/reg_test_gosdt.csv", 0.023, 0.2, 1.1,
                        run_gosdt_withoutc, reset_file=True, factor=True)


def testpyIds():
    """
    Runs the tests for the different IDS Algorithms
    Returns: Nothing

    """
    train_data = ["../res/benchmarks/train/agaricus-lepiota-Reordered/1000.csv",
                  "../res/benchmarks/train/kr-vs-kp/1000.csv",
                  "../res/benchmarks/train/adult/1000.csv",
                  "../res/benchmarks/train/spambase/1000.csv",
                  "../res/benchmarks/train/dota2TrainReordered/1000.csv"]
    test_data = ["../res/benchmarks/test/agaricus-lepiota-Reordered/1000.csv",
                 "../res/benchmarks/test/kr-vs-kp/1000.csv",
                 "../res/benchmarks/test/adult/1000.csv",
                 "../res/benchmarks/test/spambase/1000.csv",
                 "../res/benchmarks/test/dota2TrainReordered/1000.csv"]

    run_pyids(train_data, test_data, "../res/config.json", "../results/pyids_test.csv", False, alg_type="SLS")
    run_pyids(train_data, test_data, "../res/config.json", "../results/pyids_test.csv", False, alg_type="DLS")
    run_pyids(train_data, test_data, "../res/config.json", "../results/pyids_test.csv", False, alg_type="RUSM")
    run_pyids(train_data, test_data, "../res/config.json", "../results/pyids_test.csv", False, alg_type="DUSM")


def testpyIds_bin():
    """
    Runs the tests for the different IDS Algorithms
    Returns: Nothing

    """
    train_data = ["../res/benchmarks/train/agaricus-lepiota-Reordered/bin_1000.csv",
                  "../res/benchmarks/train/kr-vs-kp/bin_1000.csv",
                  "../res/benchmarks/train/adult/bin_1000.csv",
                  "../res/benchmarks/train/spambase/bin_1000.csv",
                  "../res/benchmarks/train/dota2TrainReordered/bin_1000.csv"]
    test_data = ["../res/benchmarks/test/agaricus-lepiota-Reordered/bin_1000.csv",
                 "../res/benchmarks/test/kr-vs-kp/bin_1000.csv",
                 "../res/benchmarks/test/adult/bin_1000.csv",
                 "../res/benchmarks/test/spambase/bin_1000.csv",
                 "../res/benchmarks/test/dota2TrainReordered/bin_1000.csv"]

    run_pyids(train_data, test_data, "../res/config.json", "../results/pyids_test_bin.csv", False, alg_type="SLS")
    run_pyids(train_data, test_data, "../res/config.json", "../results/pyids_test_bin.csv", False, alg_type="DLS")
    run_pyids(train_data, test_data, "../res/config.json", "../results/pyids_test_bin.csv", False, alg_type="RUSM")
    run_pyids(train_data, test_data, "../res/config.json", "../results/pyids_test_bin.csv", False, alg_type="DUSM")


def test_trainsizes():
    resultfile: str = "../results/trainsizes_test.csv"

    numbers: [str] = [50, 100, 200, 500, 1000]

    for i in range(5):
        train_data = ["../res/benchmarks/train/agaricus-lepiota-Reordered/bin_" + str(numbers[i]) + ".csv"]
        test_data = ["../res/benchmarks/test/agaricus-lepiota-Reordered/bin_" + str(numbers[i]) + ".csv"]
        train_datap = ["../res/benchmarks/train/agaricus-lepiota-Reordered/" + str(numbers[i]) + ".csv"]
        test_datap = ["../res/benchmarks/test/agaricus-lepiota-Reordered/" + str(numbers[i]) + ".csv"]

        run_gosdt_withoutc(train_data, test_data, "../res/config.json", resultfile, False, 0.005)
        run_osdt(train_data, test_data, "../res/config.json", resultfile, False, 0.005)
        run_pyids(train_datap, test_datap, "../res/config.json", resultfile, False, alg_type="SLS")
        run_pyids(train_datap, test_datap, "../res/config.json", resultfile, False, alg_type="DLS")


def test_all():
    numbers: [str] = [50, 100, 200, 500, 1000]
    datasets: [str] = ["adult", "agaricus-lepiota-Reordered", "dota2TrainReordered", "kr-vs-kp", "spambase"]

    trainsets: [str] = []
    testsets: [str] = []
    trainsetsp: [str] = []
    testsetsp: [str] = []
    gosdt_lambdas: [float] = []

    for i in range(5):
        for dataset in datasets:
            trainsets.append(["../res/benchmarks/train/" + dataset + "/bin_" + str(numbers[i]) + ".csv"])
            testsets.append(["../res/benchmarks/test/" + dataset + "/bin_" + str(numbers[i]) + ".csv"])
            trainsetsp.append(["../res/benchmarks/train/" + dataset + "/" + str(numbers[i]) + ".csv"])
            testsetsp.append(["../res/benchmarks/test/" + dataset + "/" + str(numbers[i]) + ".csv"])
            if (dataset == "adult"):
                gosdt_lambdas.append(0.075)
            if (dataset == "agaricus-lepiota-Reordered"):
                gosdt_lambdas.append(0.005)
            if (dataset == "dota2TrainReordered"):
                gosdt_lambdas.append(0.18)
            if (dataset == "kr-vs-kp"):
                gosdt_lambdas.append(0.05)
            if (dataset == "spambase"):
                gosdt_lambdas.append(0.055)

    osdt_file: str = "../results/alltests_osdt.csv"
    gosdt_file: str = "../results/alltests_gosdt.csv"
    pyids_file1: str = "../results/alltests_pyids_sls.csv"
    pyids_file2: str = "../results/alltests_pyids_dls.csv"

    for i in range(len(trainsets)):
        run_osdt(trainsets[i], testsets[i], "../res/config.json", osdt_file, False, 0.005)
        run_gosdt_withoutc(trainsets[i], testsets[i], "../res/config.json", gosdt_file, False, gosdt_lambdas[i])
        run_pyids(trainsetsp[i], testsetsp[i], "../res/config.json", pyids_file1, False, alg_type="SLS")
        run_pyids(trainsetsp[i], testsetsp[i], "../res/config.json", pyids_file2, False, alg_type="DLS")


if __name__ == '__main__':
    # testreg_osdt()
    # testreg_gosdt()
    # testpyIds()
    # testpyIds_bin()
    # test_trainsizes()
    test_all()
    # test_data = ["../res/test/monk1-train_comma.csv"]
    # test_data = ["../res/test/balance-scale_comma.csv", "../res/test/compas-binary.csv", "../res/adult/bin_500.csv"]
    # train_data = ["../res/benchmarks/train/adult/1000.csv"]
    # test_data = ["../res/benchmarks/test/adult/1000.csv"]
    # train_data = ["../res/benchmarks/train/kr-vs-kp/bin_1000.csv"]
    # test_data = ["../res/benchmarks/test/kr-vs-kp/bin_1000.csv"]
    # train_data = ["../res/benchmarks/train/dota2TrainReordered/bin_1000.csv"]
    # test_data = ["../res/benchmarks/test/dota2TrainReordered/bin_1000.csv"]
    """
    train_data = ["../res/benchmarks/train/agaricus-lepiota-Reordered/1000.csv",
                  "../res/benchmarks/train/kr-vs-kp/1000.csv",
                  "../res/benchmarks/train/adult/1000.csv",
                  "../res/benchmarks/train/spambase/1000.csv",
                  "../res/benchmarks/train/dota2TrainReordered/1000.csv"]
    test_data = ["../res/benchmarks/test/agaricus-lepiota-Reordered/1000.csv",
                 "../res/benchmarks/test/kr-vs-kp/1000.csv",
                 "../res/benchmarks/test/adult/1000.csv",
                 "../res/benchmarks/test/spambase/1000.csv",
                 "../res/benchmarks/test/dota2TrainReordered/1000.csv"]

    train_data = ["../res/benchmarks/train/agaricus-lepiota-Reordered/bin_1000.csv",
                  "../res/benchmarks/train/kr-vs-kp/bin_1000.csv",
                  "../res/benchmarks/train/adult/bin_1000.csv",
                  "../res/benchmarks/train/spambase/bin_1000.csv",
                  "../res/benchmarks/train/dota2TrainReordered/bin_1000.csv"]
    test_data = ["../res/benchmarks/test/agaricus-lepiota-Reordered/bin_1000.csv",
                 "../res/benchmarks/test/kr-vs-kp/bin_1000.csv",
                 "../res/benchmarks/test/adult/bin_1000.csv",
                 "../res/benchmarks/test/spambase/bin_1000.csv",
                 "../res/benchmarks/test/dota2TrainReordered/bin_1000.csv"]
                 """
    # train_data = ["../res/benchmarks/train/adult/bin_200.csv"]
    # test_data = ["../res/benchmarks/test/adult/bin_200.csv"]
    # train_data = ["../res/benchmarks/train/kr-vs-kp/bin_1000.csv"]
    # test_data = ["../res/benchmarks/test/kr-vs-kp/bin_1000.csv"]
    # train_data = ["../res/benchmarks/train/adult/bin_1000.csv"]
    # test_data = ["../res/benchmarks/test/adult/bin_1000.csv"]
    # run_gosdt_withc(train_data,test_data, "../res/config.json", "../results/first_result_file.csv", False, 0.1)
    # run_gosdt_withoutc(train_data,test_data, "../res/config.json", "../results/first_result_file.csv", False, 0.09)
    # run_gosdt_withc(train_data,test_data, "../res/config.json", "../results/first_result_file.csv", False, 0.08)
    # run_gosdt_withoutc(train_data,test_data, "../res/config.json", "../results/first_result_file.csv", False, 0.005)
    # run_osdt(train_data,test_data, "../res/config.json", "../results/first_result_file.csv", False, 0.1)
    # run_osdt(train_data, test_data, "../res/config.json", "../results/first_result_file.csv", False, 0.005)
    # run_osdt(train_data,test_data, "../res/config.json", "../results/first_result_file.csv", False, 0.00001)
    # run_pyids(train_data, test_data, "../res/config.json", "../results/pyids_test2.csv", False, alg_type="SLS")
    # run_pyids(train_data, test_data, "../res/config.json", "../results/pyids_test2.csv", False, alg_type="DLS")
    # run_pyids(train_data, test_data, "../res/config.json", "../results/pyids_test2.csv", False, alg_type="RUSM")
    # run_pyids(train_data, test_data, "../res/config.json", "../results/pyids_test2.csv", False, alg_type="DUSM")
    # test_regularization(train_data,test_data, "../res/config.json", "../results/reg_test_gosdtc.csv", 0.03, 0.2, 1.1, run_gosdt_withc, reset_file=True, factor=True)
    # test_regularization(train_data,test_data, "../res/config.json", "../results/reg_test_gosdtnoc_adult.csv", 0.0023, 0.2, 1.1, run_gosdt_withoutc, reset_file=True, factor=True)
    # test_regularization(train_data,test_data, "../res/config.json", "../results/reg_test.csv", 0.00000001, 0.2, 1.5, run_osdt, reset_file=True, factor=True)