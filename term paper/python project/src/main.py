from pathlib import Path

from read_write import reader
import gosdt
from model.gosdt import GOSDT
import pandas as pd

#from model.GOSDT import gosdt

if __name__ == '__main__':
    path = Path("../res/benchmarks")
    subject = Path('adult')
    size = 50
    binary_file = True
    train_file = (path / 'train' / subject / (('bin_' if binary_file else '') + str(size))).with_suffix('.csv')
    test_file = (path / 'test' / subject / (('bin_' if binary_file else '') + str(size))).with_suffix('.csv')
    data = reader.collect_data(Path('../res'))
    for name, dataset in data:
        print(name + ":")
        print(dataset)

    with open(train_file, "r") as data_file:
        data = data_file.read()

    with open("../res/config.json", "r") as config_file:
        config = config_file.read()

    print("Config:", config)
    print("Data:", data)

    gosdt.configure(config)
    result = gosdt.fit(data)

    model_file = open("../results/model.json", "w")
    model_file.write(result)
    model_file.close()

    print("Result: ", result)
    print("Time (seconds): ", gosdt.time())
    print("Iterations: ", gosdt.iterations())
    print("Graph Size: ", gosdt.size())

    dataframe = pd.DataFrame(pd.read_csv(test_file))

    X = dataframe[dataframe.columns[:-1]]
    y = dataframe[dataframe.columns[-1:]]

    """
    hyperparameters = {
        "regularization": 0.1,
        "time_limit": 3600,
        "verbose": True,
    }
    """
    with open("../res/config.json", "r") as config_file:
        hyperparameters = config_file.read()

    model = GOSDT(hyperparameters)
    model.load("../results/model.json")
    #model.fit(X, y)
    print("Execution Time: {}".format(model.time))

    model.predict(X)
    test_accuracy = model.score(X, y)
    print("Test Accuracy: {}".format(test_accuracy))

    dataframe = pd.DataFrame(pd.read_csv(train_file))
    X = dataframe[dataframe.columns[:-1]]
    y = dataframe[dataframe.columns[-1:]]
    model.predict(X)
    training_accuracy = model.score(X, y)
    print("Training Accuracy: {}".format(training_accuracy))

    print(model.tree)
    print(model.latex())
