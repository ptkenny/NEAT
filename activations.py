import numpy as np


def neat_sigmoid(x):
    return 1 / (1 + np.exp(-4.9 * x))


def tanh(x):
    return np.tanh(x)
