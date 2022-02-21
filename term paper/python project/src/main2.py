from read_write import reader
import gosdt

if __name__ == '__main__':
    data = reader.collect_data()
    for dataset in data:
        print(dataset)

    #with open("../GeneralizedOptimalSparseDecisionTrees-master/test/fixtures/binary_sepal.csv", "r") as data_file:
    #    data = data_file.read()
    with open("../res/mushroom/agaricus-lepiota-Reordered.data", "r") as data_file:
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
