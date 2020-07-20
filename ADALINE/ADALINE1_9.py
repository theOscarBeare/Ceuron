import numpy as np
from TransferFunctions import TransferFunctions

def ADALINENetwork(InputData, ErrorTolerance):
    BiasWeight = -11 + ErrorTolerance

    Zero = ZeroNeuron(InputData, BiasWeight)
    One = OneNeuron(InputData, BiasWeight)
    Two = TwoNeuron(InputData, BiasWeight)
    Three = ThreeNeuron(InputData, BiasWeight)
    Four = FourNeuron(InputData, BiasWeight)
    Five = FiveNeuron(InputData, BiasWeight)
    Six = SixNeuron(InputData, BiasWeight)
    Seven = SevenNeuron(InputData, BiasWeight)
    Eight = EightNeuron(InputData, BiasWeight)
    Nine = NineNeuron(InputData, BiasWeight)

    OutPutVector = np.array([Zero, One, Two, Three, Four, Five, Six, Seven, Eight, Nine])

    return OutPutVector


def ZeroNeuron(InputData, BiasWeight):
    weight = np.array([BiasWeight, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1])
    WeightedSum = np.dot(InputData, weight)
    Output = TransferFunctions.ReLu(WeightedSum)
    return Output

def OneNeuron(InputData, BiasWeight):
    weight = np.array([BiasWeight, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1])
    WeightedSum = np.dot(InputData, weight)
    Output = TransferFunctions.ReLu(WeightedSum)
    return Output

def TwoNeuron(InputData, BiasWeight):
    weight = np.array([BiasWeight, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1])
    WeightedSum = np.dot(InputData, weight)
    Output = TransferFunctions.ReLu(WeightedSum)
    return Output

def ThreeNeuron(InputData, BiasWeight):
    weight = np.array([BiasWeight, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1])
    WeightedSum = np.dot(InputData, weight)
    Output = TransferFunctions.ReLu(WeightedSum)
    return Output

def FourNeuron(InputData, BiasWeight):
    weight = np.array([BiasWeight, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1])
    WeightedSum = np.dot(InputData, weight)
    Output = TransferFunctions.ReLu(WeightedSum)
    return Output

def FiveNeuron(InputData, BiasWeight):
    weight = np.array([BiasWeight, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1])
    WeightedSum = np.dot(InputData, weight)
    Output = TransferFunctions.ReLu(WeightedSum)
    return Output

def SixNeuron(InputData, BiasWeight):
    weight = np.array([BiasWeight, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1])
    WeightedSum = np.dot(InputData, weight)
    Output = TransferFunctions.ReLu(WeightedSum)
    return Output

def SevenNeuron(InputData, BiasWeight):
    weight = np.array([BiasWeight,1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1])
    WeightedSum = np.dot(InputData, weight)
    Output = TransferFunctions.ReLu(WeightedSum)
    return Output

def EightNeuron(InputData, BiasWeight):
    weight = np.array([BiasWeight, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1])
    WeightedSum = np.dot(InputData, weight)
    Output = TransferFunctions.ReLu(WeightedSum)
    return Output

def NineNeuron(InputData, BiasWeight):
    weight = np.array([BiasWeight, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1])
    WeightedSum = np.dot(InputData, weight)
    Output = TransferFunctions.ReLu(WeightedSum)
    return Output
