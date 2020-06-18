import numpy as np


def act_fun(WeightedSum):
    bias = -.5
    return 1 / (1 + np.exp(WeightedSum + bias))


def tanh(WeightedSum):
    return (np.exp(WeightedSum) - np.exp(-WeightedSum)) / (np.exp(WeightedSum) + np.exp(-WeightedSum))


def ReLu(WeightedSum):
    # ReLu transfer function
    return np.where(WeightedSum > 0, WeightedSum, 0)


def softmax(WeightedSum):
    # Softmax Transfer function
    return np.exp(WeightedSum) / np.sum(np.exp(WeightedSum))


def HardLimiter(WeightedSum):
    # Hard Limiter Transfer Function
    pass
