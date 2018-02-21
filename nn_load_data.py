from nn_utils import str_column_to_float


def load_train_set():
    file = open("train_set.txt", "r")
    return parse_set(file)


def load_test_set():
    file = open("test_set.txt", "r")
    return parse_set(file)


def parse_set(file):
    expected_arr = []
    input_dataset = []

    for line in file:
        values = line.split(",")
        pixels = list(values[0])
        str_column_to_float(pixels)
        input_dataset.append(pixels)
        expected = int(values[1][0].strip())
        expected_arr.append(expected)

    return input_dataset, expected_arr
