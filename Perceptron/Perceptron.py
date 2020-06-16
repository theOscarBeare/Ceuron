import numpy as np


def perceptron(x, y):
    dataInput = x
    weights = np.random.rand(dataInput.shape[1], 4)
    y = y
    output = np.zeros(y.shape)

    weightedSum = dataInput * weights

