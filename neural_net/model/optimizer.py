import numpy as np

"""
SGD optimizer
"""


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
            if not hasattr(layer, "weight_cache"):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)

            # w, b updates with momentum
            weight_updates = (
                self.momentum * layer.weight_cache - self.current_lr * layer.dweights
            )
            layer.weight_cache = weight_updates
            bias_updates = (
                self.momentum * layer.bias_cache - self.current_lr * layer.dbiases
            )
            layer.bias_cache = bias_updates

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


"""
Adagrad optimizer
"""


class Adagrad:
    # initialize optimezer, set settings
    def __init__(self, lr=1.0, decay=0.0, epsilon=1e-7):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # call before parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):

        # create cache arrays if array doesnt have them
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        # update w, b + normalization
        layer.weights += (
            -self.current_lr
            * layer.dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )
        layer.biases += (
            -self.current_lr
            * layer.dbiases
            / (np.sqrt(layer.bias_cache) + self.epsilon)
        )

    # call after parameters update
    def post_update_params(self):
        self.iterations += 1


"""
RMSprop optimizer
"""


class RMSprop:
    # initialize optimezer, set settings
    def __init__(self, lr=1.0, decay=0.0, epsilon=1e-7, rho=0.9):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # call before parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):

        # create cache arrays if array doesnt have them
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = (
            self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        )
        layer.bias_cache = (
            self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2
        )

        # update w, b + normalization
        layer.weights += (
            -self.current_lr
            * layer.dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )
        layer.biases += (
            -self.current_lr
            * layer.dbiases
            / (np.sqrt(layer.bias_cache) + self.epsilon)
        )

    # call after parameters update
    def post_update_params(self):
        self.iterations += 1


"""
Adam optimizer
"""


class Adam:
    # initialize optimezer, set settings
    def __init__(self, lr=1.0, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # call before parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):

        # create cache arrays if array doesnt have them
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update momentum with current grads
        layer.weight_momentums = (
            self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        )
        layer.bias_momentums = (
            self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        )

        # get correct momentum
        weight_momentums_corrected = layer.weight_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )
        bias_momentums_corrected = layer.bias_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )

        # update cache
        layer.weight_cache = (
            self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        )
        layer.bias_cache = (
            self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2
        )

        # get correct cache
        weight_cache_corrected = layer.weight_cache / (
            1 - self.beta_2 ** (self.iterations + 1)
        )
        bias_cache_corrected = layer.bias_cache / (
            1 - self.beta_2 ** (self.iterations + 1)
        )

        # update w, b + normalization
        layer.weights += (
            -self.current_lr
            * weight_momentums_corrected
            / (np.sqrt(weight_cache_corrected) + self.epsilon)
        )
        layer.biases += (
            -self.current_lr
            * bias_momentums_corrected
            / (np.sqrt(bias_cache_corrected) + self.epsilon)
        )

    # call after parameters update
    def post_update_params(self):
        self.iterations += 1