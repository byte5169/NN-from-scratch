import numpy as np


class Layer:
    def __init__(self, number_inputs, number_neurons):
        # initialize w, b
        self.weights = 0.01 * np.random.randn(number_inputs, number_neurons)
        self.biases = np.zeros((1, number_neurons))

    # forward pass
    def forward(self, inputs):
        # calculate output from i, w, b
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # backward pass
    def backward(self, dvalues):
        # gradient on w, b
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # gradient on inputs
        self.dinputs = np.dot(dvalues, self.weights.T)