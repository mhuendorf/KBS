from pathlib import Path

from read_write import reader
import gosdt

#from model.GOSDT import gosdt

if __name__ == '__main__':
    data = reader.collect_data(Path('../res'))
    for name, dataset in data:
        print(name + ":")
        print(dataset)

    with open("../res/test/monk1-train_comma.csv", "r") as data_file:
        data = data_file.read()

    with open("../res/config.json", "r") as config_file:
        config = config_file.read()

    print("Config:", config)
    print("Data:", data)

    gosdt.configure(config)
    result = gosdt.fit(data)

    print("Result: ", result)
    print("Time (seconds): ", gosdt.time())
    print("Iterations: ", gosdt.iterations())
    print("Graph Size: ", gosdt.size())
