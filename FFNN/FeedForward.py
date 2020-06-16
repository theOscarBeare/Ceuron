import numpy as np


def feed_forward(inputs, outputs, weights):
    pre_hidden = np.dot(inputs, weights[0]) + weights[1]
    hidden = 1 / (1 + np.exp(-pre_hidden))
    pred_out = np.dot(hidden, weights[2]) + weights[3]
    squared_error = (np.square(pred_out - outputs))
    return squared_error
