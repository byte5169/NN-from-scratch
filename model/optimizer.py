import numpy as np
from numpy.core.numeric import zeros_like

### SGD optimizer
class SGD:
    # initialize optimezer, set settings
    def __init__(self, lr=1.0, decay=0.0, momentum=0.0):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # call before parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):

        # when we use momentum
        if self.momentum:
            # create momentum arrays if array doesnt have them
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # w, b updates with momentum
            weight_updates = (
                self.momentum * layer.weight_momentums
                - self.current_lr * layer.dweights
            )
            layer.weight_momentums = weight_updates
            bias_updates = (
                self.momentum * layer.bias_momentums - self.current_lr * layer.dbiases
            )
            layer.bias_momentums = bias_updates

        # before momentum updates
        else:
            weight_updates = -self.current_lr * layer.dweights
            bias_updates = -self.current_lr * layer.dbiases

        # update w, b using either vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # call after parameters update
    def post_update_params(self):
        self.iterations += 1
