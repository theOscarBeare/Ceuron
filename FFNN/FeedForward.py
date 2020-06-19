import numpy as np
from Perceptron import PerceptronBuild


def upDateWeights(squareErrorsMatrix, weights, row):
    learningRate = [0.35]

    verticalSum = squareErrorsMatrix.sum(axis=0)

    print(verticalSum)
    dw = learningRate * verticalSum
    print(dw)
    newWeights = weights * dw
    print(newWeights)

    return newWeights


def feed_forward(TF, dataInput, targets):
    weights = np.array([])
    squareErrorsMatrix = np.array([[]])
    squareErrors = np.array([])
    newWeights = np.array([])

    for row in dataInput:
        weights = np.random.rand(4, 3)

    for row in dataInput:

        perceptronDataInput = np.array(dataInput[row])
        perceptronWeightsInput = np.array(weights[row])

        perceptronOutput = PerceptronBuild.perceptron(perceptronDataInput, TF, perceptronWeightsInput)

        if (targets[row] - perceptronOutput).any:
            squareErrorsMatrix = dataInput[row]
        elif (targets[row] - perceptronOutput) <= 0:
            squareErrorsMatrix = -dataInput[row]

    weightsRow = 0

    while weightsRow < 3:
        newWeights[weightsRow] = upDateWeights(squareErrorsMatrix, weights[weightsRow], weightsRow)
        weightsRow = weightsRow +1

    weights = newWeights

    return weights
