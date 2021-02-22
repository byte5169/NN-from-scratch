import numpy as np

### Accuracy class
class Accuracy:
    # calculating accuracy using layer output, targets
    def calc(self, output, y):
        preds = np.argmax(output, axis=1)
        # if targets one-hot enc - covert them
        if len(y.shape)==2:
            y = np.argmax(y, axis=1)
        acc = np.mean(preds==y)

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