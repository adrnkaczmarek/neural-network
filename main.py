from nn_load_data import load_test_set, load_train_set
from neural_network import NeuralNetwork
import numpy as np


if __name__ == '__main__':
    nn = NeuralNetwork(64, 3, 3)
    train_set, expected = load_train_set()
    test_set, _ = load_test_set()

    for _ in range(50):
        for index in range(len(train_set)):
            input_values = np.array(train_set[index])
            expected_vector = np.zeros(3)
            expected_vector[expected[index] - 1] = 1
            nn.train(input_values, expected_vector)

    for index in range(len(test_set)):
        print(nn.predict(test_set[index]))
