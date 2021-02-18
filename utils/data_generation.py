# the base of this function was taken and a bit modified from
# https://cs231n.github.io/neural-networks-case-study/

import matplotlib.pyplot as plt
import numpy as np

# set seed to reproduse results across runs
np.random.seed(42)

# func to generate the data
def data_generation(number=100, classes=3):
    number = number  # number of points per class
    classes = classes  # number of classes
    dimension = 2  # dimensionality
    X = np.zeros(
        (number * classes, dimension)
    )  # data matrix (each row = single example)
    y = np.zeros(number * classes, dtype="uint8")  # class labels
    for j in range(classes):
        ix = range(number * j, number * (j + 1))
        r = np.linspace(0.0, 1, number)  # radius
        t = (
            np.linspace(j * 4, (j + 1) * 4, number) + np.random.randn(number) * 0.2
        )  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y


# func to visualize the data
def data_visualize(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()


# working example
# X, y = data_generation()
# data_visualize(X, y)
