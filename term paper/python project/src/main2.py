from read_write import reader
import gosdt
import pandas as pd
from model.gosdt import GOSDT

if __name__ == '__main__':
    #data = reader.collect_data()
    #for dataset in data:
    #    print(dataset)
    #data = reader.collect_data('../res')
    #for name, dataset in data:
    #    print(name + ":")
    #    print(dataset)

    with open("../res/config.json", "r") as config_file:
        config = config_file.read()
    gosdt.configure(config)

    dataframe = pd.DataFrame(pd.read_csv("../res/benchmarks/adult/50.csv"))

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
    model.fit(X, y)
    print("Execution Time: {}".format(model.time))

    prediction = model.predict(X)
    training_accuracy = model.score(X, y)
    print("Training Accuracy: {}".format(training_accuracy))
    print(model.tree)