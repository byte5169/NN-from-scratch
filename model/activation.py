import numpy as np

# ReLU activation function
class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Softmax activation function
class Softmax:
    def forward(self, inputs):
        ex_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalizing probabilities
        self.output = ex_val/np.sum(ex_val, axis=1, keepdims=True)
