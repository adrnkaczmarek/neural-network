import numpy as np


def str_column_to_float(row):
    for i in range(len(row)):
        row[i] = float(row[i].strip())


def sigmoid(x):
    return 1./(1. + np.exp(-x))


def add_bias(input_data):
    return np.insert(input_data, 0, 1)


def derivative(x):
    return x * (1.0 - x)


def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))
