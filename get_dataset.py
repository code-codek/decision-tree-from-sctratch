def get_dataset(file_name):
    f = open(file_name, "r")
    dataset = []
    
    for line in f:
        dataset.append(line.rstrip().split(","))

    return dataset