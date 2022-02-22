from read_write import reader
import gosdt
import pandas as pd

if __name__ == '__main__':
    #data = reader.collect_data()
    #for dataset in data:
    #    print(dataset)

    #with open("../GeneralizedOptimalSparseDecisionTrees-master/test/fixtures/binary_sepal.csv", "r") as data_file:
    #    data = data_file.read()
    #with open("../res/spam/spambase.data", "r") as data_file:
    #    data = data_file.read()

    #data = pd.read_csv("../res/spam/spambase.data")
    data = pd.read_csv("../res/mushroom/agaricus-lepiota-Reordered.data")

    with open("../res/config.json", "r") as config_file:
        config = config_file.read()

    print("Config:", config)
    print("Data:", data)

    datastring = ""

    for col in data.columns:
        print(col)
        #data[col] = pd.cut(data[col],5).astype(str)
        data[col] = pd.cut(data[col], 5).astype(str)

    for row in range(1,len(data.index)):
        for col in data.columns:
            datastring += data[col][row].replace(' ','').replace(',','-') + ","
        datastring = datastring[:-1]
        datastring += "\n"

    print("Data:", data)
    #"""
    gosdt.configure(config)
    #datastring2 = data.to_string()
    print(datastring)
    result = gosdt.fit(datastring)

    print("Result: ", result)
    print("Time (seconds): ", gosdt.time())
    print("Iterations: ", gosdt.iterations())
    print("Graph Size: ", gosdt.size())
    #"""
