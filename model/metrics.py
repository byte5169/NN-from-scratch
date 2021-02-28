from model.activation import Softmax
import numpy as np

### Accuracy class
class Accuracy:
    # calculating accuracy using layer output, targets
    def calc(self, output, y):
        preds = np.argmax(output, axis=1)
        # if targets one-hot enc - covert them
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        acc = np.mean(preds == y)

        return acc


### General Loss class
class Loss:
    # takes model output, true values and calculates
    # data losses
    def calc(self, output, y):
        loss_sample = self.forward(output, y)
        loss_mean = np.mean(loss_sample)

        return loss_mean


### Categorical Cross Entropy loss class
class CatCrossEntropy(Loss):
    # forwad pass
    def forward(self, y_pred, y_true):
        # number of samples in a batch
        batch_samples = len(y_pred)
        # clip data to prevent div by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # probabilities for categorical values
        if len(y_true.shape) == 1:
            probs = y_pred_clipped[range(batch_samples), y_true]

        # probs for one-hot enc values
        elif len(y_true.shape) == 2:
            probs = np.sum(y_pred_clipped * y_true, axis=1)

        # losses
        neg_log = -np.log(probs)

        return neg_log

    # backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # number of labels in a sample
        labels = len(dvalues[0])

        # convert to one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # calc gradient
        self.dinputs = -y_true / dvalues
        # normalize gradient
        self.dinputs = self.dinputs / samples


### Combine Softmax and Categorical crossentropy for faster calculations
class Softmax_CatCrossEntropy:
    # create activation and loss functions
    def __init__(self):
        self.activation = Softmax()
        self.loss = CatCrossEntropy()

    # forward pass
    def forward(self, inputs, y_true):
        # output laers activation
        self.activation.forward(inputs)
        # output
        self.output = self.activation.output
        # loss
        return self.loss.calc(self.output, y_true)

    # backward pass
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)

        # if labels one-hot => discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # copy
        self.dinputs = dvalues.copy()
        # calc gradient
        self.dinputs[range(samples), y_true] -= 1
        # normalize gradient
        self.dinputs = self.dinputs / samples
