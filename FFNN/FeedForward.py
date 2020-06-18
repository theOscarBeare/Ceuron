import numpy as np
from Perceptron import PerceptronBuild


def upDateWeights(squareErrors, weights):
    learningRate = [0.35, 0.35, 0.35]
    verticalSum = squareErrors.sum(axis=0)

    dw = learningRate * verticalSum
    newWeights = weights * dw

    return newWeights


def feed_forward(TF, dataInput, targets):
    weights = [[]]
    squareErrors = [[]]
    squareErrorsList = []

    for row in dataInput:
        weights = np.random.rand(dataInput[row], 4)
    print(weights)

    for row in dataInput:

        perceptronOutput = PerceptronBuild.perceptron(dataInput[row], TF, weights[row])

        if (targets(row) - perceptronOutput) > 0:
            squareErrorsList = dataInput[row]
        elif (targets(row) - perceptronOutput) <= 0:
            squareErrorsList = -dataInput[row]

        squareErrors[row] = squareErrorsList

    # reshaping the matrix to allow for the vertical sum calculation
    squareErrors = np.squareErrors.reshape(4, 3)
    newWeights = upDateWeights(squareErrors, weights)

    weights = newWeights

    return weights
