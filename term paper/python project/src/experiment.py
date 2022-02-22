#Author: Gideon Vogt
import gosdt
from osdt import bbound
from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
import os.path
#import sys
#sys.path.append('../GeneralizedOptimalSparseDecisionTrees-master/python')
from model.gosdt import GOSDT

def runGosdt(data_csv_paths:[str], config_file_path:str, result_file_path:str, reset_file:bool = False):
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

    if((not file_exists) or reset_file):
        resultFile = open(result_file_path, "w")
    else:
        resultFile = open(result_file_path, "a")

    # read and apply config
    with open("../res/config.json", "r") as config_file:
        config = config_file.read()
    gosdt.configure(config)

    # solve datasets given
    for dataCsvPath in data_csv_paths:
        # read in data
        with open("../GeneralizedOptimalSparseDecisionTrees-master/test/fixtures/binary_sepal.csv", "r") as data_file:
            data = data_file.read()
        # execute
        result = gosdt.fit(data)


