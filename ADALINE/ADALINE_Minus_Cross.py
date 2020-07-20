import numpy as np
from TransferFunctions import TransferFunctions

def ADALINENetwork(InputData, ErrorTolerance):
    biasWeight = -11 + ErrorTolerance

    cross = crossNeuron(InputData, biasWeight)
    minus = minusNeuron(InputData, biasWeight)

    outputVector = np.array([cross, minus])

    return outputVector


def crossNeuron(inputData, BiasWeight):
    weight = np.array([BiasWeight, -1, 1, -1, 1, 1, 1, -1, 1, -1])
    WeightedSum = np.dot(inputData, weight)
    output = TransferFunctions.ReLu(WeightedSum)
    return output


def minusNeuron(inputData, BiasWeight):
    weight = np.array([BiasWeight, -1, -1, -1, 1, 1, 1, -1, -1, -1])
    WeightedSum = np.dot(inputData, weight)
    output = TransferFunctions.ReLu(WeightedSum)
    return output
