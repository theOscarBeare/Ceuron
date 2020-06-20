from TransferFunctions import TransferFunctions
import numpy as np


def perceptron(dataInput, TF, weights):
    output = int
    weightedSum = np.dot(dataInput, weights)

    if TF == "softmax":
        output = TransferFunctions.softmax(weightedSum)
    elif TF == "ReLu":
        output = TransferFunctions.ReLu(weightedSum)
    elif TF == "tanh":
        output = TransferFunctions.tanh(weightedSum)
    elif TF == "act_fun":
        output = TransferFunctions.act_fun(weightedSum)
    elif TF == "hardlimiter":
        output = TransferFunctions.HardLimiter(weightedSum)
    else:
        print("The Transfer function that you have entered is either not implemented or has been entered incorrectly"
              "the current functions are: softmax, ReLu, tanh, act_fun, hardlimiter")

    # prints the Matrix output of the instances
    print(output)

    return output
