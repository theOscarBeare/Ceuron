import numpy as np


class TransferFunctions:

    def __init__(self, TF, x):
        self.TransferFunction = TF
        self.WeightedSum = x

    def act_fun(self):
        bias = -.5
        return 1 / (1 + np.exp(-self.WeightedSum + bias))

    def tanh(self):
        return (np.exp(self.WeightedSum) - np.exp(-self.WeightedSum)) / (np.exp(self.WeightedSum) + np.exp(-self.WeightedSum))

    def ReLu(self):
        # ReLu transfer function
        return np.where(self.WeightedSum > 0, self.WeightedSum, 0)

    def softmax(self):
        # Softmax Transfer function
        return np.exp(self.WeightedSum) / np.sum(np.exp(self.WeightedSum))

    def HardLimiter(self):
        # Hard Limiter Transfer Function
        pass
