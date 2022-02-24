from pathlib import Path

from read_write import reader
import gosdt
from model.gosdt import GOSDT
import pandas as pd

#from model.GOSDT import gosdt

if __name__ == '__main__':
    res_file = "../res/benchmarks/adult/50.csv"
    data = reader.collect_data(Path('../res'))
    for name, dataset in data:
        print(name + ":")
        print(dataset)

    with open(res_file, "r") as data_file:
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

    dataframe = pd.DataFrame(pd.read_csv(res_file))

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

    prediction = model.predict(X)
    training_accuracy = model.score(X, y)
    print("Training Accuracy: {}".format(training_accuracy))
    print(model.tree)
    print(model.latex())
