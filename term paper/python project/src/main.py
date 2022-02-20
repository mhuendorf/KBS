from read_write import reader

if __name__ == '__main__':
    data = reader.collect_data()
    for dataset in data:
        print(dataset)
