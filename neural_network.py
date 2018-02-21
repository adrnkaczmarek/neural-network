import numpy as np
from nn_utils import sigmoid, add_bias, derivative


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        epsilon = 0.9
        self.first_layer_weights = np.random.rand(input_size + 1, hidden_size) * (2 * epsilon) - epsilon
        self.second_layer_weights = np.random.rand(hidden_size + 1, output_size) * (2 * epsilon) - epsilon
        self.learning_rate = 0.2
        self.input_size = input_size

    def __forward(self, input_data):
        input_data = add_bias(input_data)
        z1 = input_data.dot(self.first_layer_weights)
        hidden_layer = sigmoid(z1)
        hidden_layer = add_bias(hidden_layer)
        z2 = hidden_layer.dot(self.second_layer_weights)
        output_layer = sigmoid(z2)
        return hidden_layer, output_layer

    def __backward(self, input_data, hidden_layer, output_layer, expected):
        output_error = (expected - output_layer) * derivative(output_layer)
        self.second_layer_weights = self.second_layer_weights + self.learning_rate * np.outer(hidden_layer, output_error)
        hidden_error = (self.second_layer_weights[1:].dot(output_error)) * derivative(output_layer)
        self.first_layer_weights = self.first_layer_weights + self.learning_rate * np.outer(add_bias(input_data), hidden_error)

    def predict(self, row):
        _, output_layer = self.__forward(row)
        return np.argmax(output_layer) + 1

    def train(self, input_data, expected):
        if len(input_data) == self.input_size:
            hidden_layer, output_layer = self.__forward(input_data)
            self.__backward(input_data, hidden_layer, output_layer, expected)
        else:
            print("Input data size is equal " + str(len(input_data)) +
                  ". Instead it should has length equal " + self.input_size)
