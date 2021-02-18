# the base of this function was taken and a bit modified from
# https://cs231n.github.io/neural-networks-case-study/

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# func to generate the data
def data_generation(N=100, K=3):
    N = N  # number of points per class
    D = 2  # dimensionality
    K = K  # number of classes
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype="uint8")  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
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
