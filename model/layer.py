import numpy as np

# set seed to reproduse results across runs
np.random.seed(42)


class Layer:
    def __init__(self, number_inputs, number_neurons):
        # initialize w, b
        self.weights = np.random.randn(number_inputs, number_neurons)
        self.biases = np.zeros((1, number_neurons))

    def forward(self, inputs):
        # calculate output from i, w, b
        self.output = np.dot(inputs, self.weights) + self.biases
