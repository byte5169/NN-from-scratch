import numpy as np

"""
ReLU activation function
"""


class ReLU:
    # forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # backward pass
    def backward(self, dvalues):
        # make a copy of original value as we modify it
        self.dinputs = dvalues.copy()
        # zero gradient where inout is negative
        self.dinputs[self.inputs <= 0] = 0


"""
Softmax activation function
"""


class Softmax:
    # forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # unnormalized probs
        ex_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalizing probabilities
        probs = ex_val / np.sum(ex_val, axis=1, keepdims=True)
        self.output = probs

    # backward pass
    def backward(self, dvalues):
        # uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # enumerate gradients and outputs
        for i, (one_output, one_dvalue) in enumerate(zip(self.output, dvalues)):
            # flatten array
            one_output = one_output.reshape(-1, 1)
            # jacobian matrix
            jac_matrix = np.diagflat(one_output) - np.dot(one_output, one_output.T)
            # calc sample gradient and add it to array of sample gradients
            self.dinputs[i] = np.dot(jac_matrix, one_dvalue)